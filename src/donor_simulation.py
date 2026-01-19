from openai import OpenAI
import json
from src.agents import Agent
from src.utils import (
    get_client_config,
    get_llm_provider,
    reset_llm_error_counts,
    print_llm_error_summary,
)
from concurrent.futures import ThreadPoolExecutor


class DonorSimulationData:
    """Centralized state management for donor game simulations."""

    def __init__(self):
        self.agents = {}
        self.conversation_history = []
        self.game_data = {}
        self.task_order = None
        self.myth_visibility = "all"  # "all" or "recent_partners"
        self.myth_partner_rounds = 1  # How many rounds back to look for partners

    def add_agent(self, agent_id, agent):
        self.agents[agent_id] = agent

    def get_agent_messages(self, agent_id):
        return self.agents[agent_id].messages

    def save_state(self, filepath):
        state = {
            "conversation_history": self.conversation_history,
            "game_data": self.game_data,
            "task_order": self.task_order,
            "myth_visibility": self.myth_visibility,
            "myth_partner_rounds": self.myth_partner_rounds,
            "agents": {
                agent_id: {
                    "agent_id": agent.agent_id,
                    "model": agent.model,
                    "memory_capacity": agent.memory_capacity,
                    "initial_bias": agent.initial_bias,
                    "system_prompt": agent.system_prompt,
                    "messages": agent.messages
                }
                for agent_id, agent in self.agents.items()
            }
        }
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)


def get_agent_reasoning(agent, prompt):
    """Get reasoning response from agent."""
    return agent.respond(prompt)


def get_agent_donation(agent, prompt):
    """Get donation amount response from agent."""
    return agent.respond(prompt)


def run_donor_simulation(
    game,
    model,
    temperature,
    num_turns,
    memory_capacity,
    agent_biases,
    myth_writer=None,
    task_order=None,
    myth_visibility="all",
    myth_partner_rounds=1,
):
    """
    Run a donor game simulation with parallel decision-making.

    Args:
        game: DonorGame instance
        model: LLM model identifier
        temperature: LLM temperature
        num_turns: Number of rounds to play
        memory_capacity: Agent memory capacity
        agent_biases: List of biases for each agent
        myth_writer: Optional MythWriter for myth generation
        task_order: List of tasks per round, e.g., ["game", "myth"] or ["game"]
        myth_visibility: "all" or "recent_partners" - whose myths agents can see
        myth_partner_rounds: How many rounds back to look for partners (for recent_partners mode)
    """
    if task_order is None:
        task_order = ["game"]

    # Reset per-run LLM error counters
    reset_llm_error_counts()

    # Create client for detected/configured provider
    provider = get_llm_provider()
    client_config = get_client_config(provider)
    client = OpenAI(**client_config)

    sim_data = DonorSimulationData()
    sim_data.task_order = task_order
    sim_data.myth_visibility = myth_visibility
    sim_data.myth_partner_rounds = myth_partner_rounds

    # Initialize agents
    num_agents = game.population_size
    for i in range(num_agents):
        agent_id = f"Agent_{i+1}"
        bias = agent_biases[i] if agent_biases and i < len(agent_biases) else None
        agent = Agent(
            agent_id, model, temperature, client,
            memory_capacity=memory_capacity, initial_bias=bias
        )
        system_prompt = game.get_system_prompt(agent_id, agent)
        agent.system_prompt = system_prompt
        agent.messages.append({"role": "system", "content": system_prompt})
        sim_data.add_agent(agent_id, agent)

    # Initialize game data
    game.initialize_game_data(sim_data)

    print("\n" + "=" * 80)
    print("DONOR GAME SIMULATION")
    print("=" * 80)
    print(f"LLM Provider: {provider}")
    print(f"Model: {model}")
    print(f"Population size: {num_agents}")
    print(f"Initial resources: {game.initial_resources}")
    print(f"Donation multiplier: {game.donation_multiplier}x")
    print(f"Context depth: {game.context_depth}, width: {game.context_width}")
    print(f"Number of rounds: {num_turns}")
    print(f"Task order: {task_order}")
    if myth_writer:
        print(f"Myth visibility: {myth_visibility}")
        if myth_visibility == "recent_partners":
            print(f"Myth partner rounds: {myth_partner_rounds}")
    print("=" * 80)

    # Main simulation loop
    for turn in range(1, num_turns + 1):
        print("\n" + "=" * 80)
        print(f"ROUND {turn}")
        print("=" * 80)

        # Pre-create conversation history entry for this round
        round_entry = {
            "round": turn,
            "pairings": None,
            "donations": None,
            "balances": None,
            "actions": None,
            "myths": {},
        }
        sim_data.conversation_history.append(round_entry)

        # Execute tasks in specified order
        for task in task_order:
            if task == "game":
                _execute_game_round(game, turn, sim_data)

            elif task == "myth" and myth_writer:
                _execute_myth_round(
                    myth_writer, turn, sim_data,
                    myth_visibility, myth_partner_rounds
                )

        # Print turn summary if game was played
        if "game" in task_order:
            game.print_turn_summary(turn, {}, sim_data)

        # Print myths if written
        if "myth" in task_order and myth_writer:
            _print_myths(turn, sim_data)

    # Print final summary
    if "game" in task_order:
        game.print_game_summary(sim_data)

    print_llm_error_summary()
    return sim_data


def _execute_game_round(game, turn, sim_data):
    """Execute one round of the donor game with parallel decisions."""
    print("\n--- GAME PHASE ---")

    # Generate pairings for this round
    pairings = game.generate_pairings(turn, sim_data)
    sim_data.game_data["current_pairings"] = pairings

    print(f"\nPairings this round:")
    for agent_a, agent_b in pairings:
        print(f"  {agent_a} ↔ {agent_b}")

    # Build mapping of agent -> partner
    partner_map = {}
    for agent_a, agent_b in pairings:
        partner_map[agent_a] = agent_b
        partner_map[agent_b] = agent_a

    # Phase 1: Get reasoning from all agents in parallel
    print("\nPhase 1: Collecting reasoning from all agents...")
    reasoning_prompts = {}
    for agent_id in sim_data.agents.keys():
        partner_id = partner_map[agent_id]
        reasoning_prompts[agent_id] = game.get_reasoning_prompt(
            agent_id, partner_id, turn, sim_data
        )

    reasoning_responses = {}
    with ThreadPoolExecutor(max_workers=len(sim_data.agents)) as executor:
        futures = {
            agent_id: executor.submit(
                get_agent_reasoning,
                sim_data.agents[agent_id],
                reasoning_prompts[agent_id]
            )
            for agent_id in sim_data.agents.keys()
        }

        for agent_id, future in futures.items():
            reasoning_responses[agent_id] = future.result()
            print(f"\n{agent_id} reasoning (paired with {partner_map[agent_id]}):")
            print(f"  {reasoning_responses[agent_id][:200]}...")

    # Phase 2: Get donation amounts from all agents in parallel
    print("\nPhase 2: Collecting donation amounts from all agents...")
    donation_prompts = {}
    for agent_id in sim_data.agents.keys():
        partner_id = partner_map[agent_id]
        donation_prompts[agent_id] = game.get_donation_prompt(
            agent_id, partner_id, turn, sim_data
        )

    donation_responses = {}
    with ThreadPoolExecutor(max_workers=len(sim_data.agents)) as executor:
        futures = {
            agent_id: executor.submit(
                get_agent_donation,
                sim_data.agents[agent_id],
                donation_prompts[agent_id]
            )
            for agent_id in sim_data.agents.keys()
        }

        for agent_id, future in futures.items():
            donation_responses[agent_id] = future.result()
            print(f"\n{agent_id} donation response: {donation_responses[agent_id]}")

    # Extract donation amounts and build response data
    agent_responses = {}
    for agent_id in sim_data.agents.keys():
        partner_id = partner_map[agent_id]
        balance = sim_data.game_data["balances"][agent_id]
        donation_amount = game.extract_donation_amount(
            donation_responses[agent_id], balance
        )

        agent_responses[agent_id] = {
            "reasoning": reasoning_responses[agent_id],
            "donation_response": donation_responses[agent_id],
            "donation_amount": donation_amount,
            "partner_id": partner_id,
        }

        print(f"{agent_id} donates {donation_amount:.1f} to {partner_id}")

    # Process the turn (update balances, record history)
    game.process_turn(turn, agent_responses, sim_data)


def _execute_myth_round(myth_writer, turn, sim_data, myth_visibility, myth_partner_rounds):
    """Execute myth writing for all agents in parallel."""
    print("\n--- MYTH PHASE ---")

    agent_ids = list(sim_data.agents.keys())

    # Get recent partners for each agent if using recent_partners visibility
    recent_partners = {}
    if myth_visibility == "recent_partners":
        recent_partners = _get_recent_partners(sim_data, myth_partner_rounds)

    # Generate prompts for all agents
    prompts = {}
    for agent_id in agent_ids:
        if turn == 1:
            prompts[agent_id] = myth_writer.get_myth_prompt_round_1(
                agent_id, turn, sim_data
            )
        else:
            # Get visible myths based on visibility setting
            visible_myths = _get_visible_myths(
                agent_id, turn, sim_data,
                myth_visibility, recent_partners
            )
            prompts[agent_id] = _get_myth_prompt_with_visibility(
                myth_writer, agent_id, turn, sim_data, visible_myths
            )

    # Parallelize LLM calls
    agent_myths = {}
    with ThreadPoolExecutor(max_workers=len(agent_ids)) as executor:
        futures = {
            agent_id: executor.submit(
                sim_data.agents[agent_id].respond, prompts[agent_id]
            )
            for agent_id in agent_ids
        }

        for agent_id, future in futures.items():
            agent_myths[agent_id] = future.result()
            print(f"\n{agent_id} myth prompt:\n{prompts[agent_id][:200]}...")
            print(f"\n{agent_id} myth:\n{agent_myths[agent_id][:200]}...")

    # Store myths in conversation history
    myth_writer.process_myths(turn, agent_myths, sim_data)


def _get_recent_partners(sim_data, num_rounds):
    """
    Get the recent partners for each agent over the last N rounds.

    Returns dict: {agent_id: set of partner_ids}
    """
    recent_partners = {agent_id: set() for agent_id in sim_data.agents.keys()}

    pairings_history = sim_data.game_data.get("pairings_history", [])

    # Look at the last num_rounds of pairings
    recent_pairings = pairings_history[-num_rounds:] if pairings_history else []

    for pairing_entry in recent_pairings:
        for agent_a, agent_b in pairing_entry.get("pairings", []):
            recent_partners[agent_a].add(agent_b)
            recent_partners[agent_b].add(agent_a)

    return recent_partners


def _get_visible_myths(agent_id, turn, sim_data, myth_visibility, recent_partners):
    """
    Get the myths visible to an agent based on visibility settings.

    Returns dict: {other_agent_id: myth_text}
    """
    visible_myths = {}

    # Get myths from previous round
    prev_round_entry = None
    for entry in sim_data.conversation_history:
        if entry["round"] == turn - 1 and "myths" in entry:
            prev_round_entry = entry
            break

    if not prev_round_entry or "myths" not in prev_round_entry:
        return visible_myths

    prev_myths = prev_round_entry["myths"]

    if myth_visibility == "all":
        # Agent sees all other agents' myths
        for other_id, myth in prev_myths.items():
            if other_id != agent_id:
                visible_myths[other_id] = myth

    elif myth_visibility == "recent_partners":
        # Agent only sees myths from recent partners
        partners = recent_partners.get(agent_id, set())
        for other_id, myth in prev_myths.items():
            if other_id != agent_id and other_id in partners:
                visible_myths[other_id] = myth

    return visible_myths


def _get_myth_prompt_with_visibility(myth_writer, agent_id, turn, sim_data, visible_myths):
    """
    Generate myth prompt with appropriate visibility of other agents' myths.
    """
    # Get this agent's own previous myth
    own_myth = None
    for entry in sim_data.conversation_history:
        if entry["round"] == turn - 1 and "myths" in entry:
            own_myth = entry["myths"].get(agent_id)
            break

    if not own_myth:
        # Fall back to round 1 prompt if no previous myth
        return myth_writer.get_myth_prompt_round_1(agent_id, turn, sim_data)

    # Build prompt with visible myths
    if visible_myths:
        other_myths_text = "\n\n".join([
            f"Myth from {other_id}:\n{myth}"
            for other_id, myth in visible_myths.items()
        ])

        prompt = f"""Here is the myth you wrote in the previous round:
{own_myth}

Here are myths from other agents that you can see:
{other_myths_text}

Write your own myth. Use your previous myth as inspiration, but adapt it in your own way.
Write 200 words. Format exactly:
Myth: [your story here]."""

    else:
        # No visible myths from others - only show own myth
        prompt = f"""Here is the myth you wrote in the previous round:
{own_myth}

Write your own myth. Use your previous myth as inspiration, but adapt it in your own way.
Write 200 words. Format exactly:
Myth: [your story here]."""

    return prompt


def _print_myths(turn, sim_data):
    """Print myths written this round."""
    for entry in sim_data.conversation_history:
        if entry["round"] == turn and entry.get("myths"):
            print(f"\n{'~' * 80}")
            print("MYTHS WRITTEN THIS ROUND:")
            print(f"{'~' * 80}")
            for agent_id, myth in entry["myths"].items():
                print(f"\n{agent_id}:")
                print(myth[:500] + "..." if len(myth) > 500 else myth)
                print("-" * 40)
            break
