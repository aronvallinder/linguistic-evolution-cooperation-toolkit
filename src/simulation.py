from together import Together
from openai import OpenAI
import json
from src.agents import Agent
from src.utils import (
    print_simulation_header,
    get_openrouter_api_key,
    reset_llm_error_counts,
    print_llm_error_summary,
)
from concurrent.futures import ThreadPoolExecutor
class SimulationData:
    """Centralized state management for multi-agent conversations"""

    def __init__(self):
        self.agents = {}
        self.conversation_history = []
        self.game_data = {}
        self.task_order = None  # Store task order used in simulation

    def add_agent(self, agent_id, agent):
        self.agents[agent_id] = agent

    def get_agent_messages(self, agent_id):
        return self.agents[agent_id].messages

    def save_state(self, filepath):
        state = {
            "conversation_history": self.conversation_history,
            "game_data": self.game_data,
            "task_order": self.task_order, # Include task_order in saved state
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

def run_simulation(game, model, temperature, num_turns, num_agents, memory_capacity, agent_biases, myth_writer, task_order=["game", "myth"]):
    """
    Run a multi-agent simulation with any game.
    Now supports sequential moves within a turn.
    Args:
    task_order: List of tasks to execute in order. Options: "game", "myth"
                Examples: ["game"], ["myth"], ["game", "myth"], ["myth", "game"]
    """
    # Reset per-run LLM error counters (useful for batch runs in a single process).
    reset_llm_error_counts()

    # If you want to run via Together instead of OpenRouter, use:
    # from src.utils import get_together_api_key
    # client = Together(api_key=get_together_api_key())
    client = OpenAI(api_key=get_openrouter_api_key(), base_url="https://openrouter.ai/api/v1")
    sim_data = SimulationData()
    sim_data.task_order = task_order  # Store task_order in sim_data

    # Initialize agents
    for i in range(num_agents):
        agent_id = f"Agent_{i+1}"
        bias = agent_biases[i] if agent_biases and i < len(agent_biases) else None
        agent = Agent(agent_id, model, temperature, client, memory_capacity=memory_capacity, initial_bias=bias)
        system_prompt = game.get_system_prompt(agent_id, agent)
        agent.system_prompt = system_prompt
        agent.messages.append({"role": "system", "content": system_prompt})
        sim_data.add_agent(agent_id, agent)

    print_simulation_header(game, num_turns, num_agents, memory_capacity, agent_biases)
    last_responses = {}

    # Main simulation loop
    for turn in range(1, num_turns + 1):
        print("\n" + "=" * 80)
        print(f"ROUND {turn}")
        print("=" * 80)

        # Display role assignments for this round
        roles = game.get_roles_for_round(turn)
        move_order = game.get_move_order(turn, sim_data)

        print(f"Roles this round: {roles['investor']} = INVESTOR, {roles['trustee']} = TRUSTEE")
        print(f"Move order this round: {move_order}")
        # Pre-create complete conversation_history entry for this round with all fields
        round_entry = {
            "round": turn,
            "roles": {
                roles["investor"]: "investor",
                roles["trustee"]: "trustee"
            },
            "sent": None,
            "received": None,
            "returned": None,
            "investor_payoff": None,
            "trustee_payoff": None,
            "balances": None,
            "actions": None,
            "myths": {}
        }
        sim_data.conversation_history.append(round_entry)

        agent_responses = {}
        agent_myths = {}

        # Execute tasks in specified order
        for task in task_order:
            if task == "game":
                # PHASE 1: GAME PLAY
                print("\n--- PHASE 1: GAME PLAY ---")
                
                # Sequential moves
                #move_order = game.get_move_order(turn, sim_data)
                
                for agent_id in move_order:
                    agent = sim_data.agents[agent_id]
                    current_role = roles["investor"] if agent_id == roles["investor"] else roles["trustee"]
                    role_name = "Investor" if current_role == roles["investor"] else "Trustee"
                    
                    if turn == 1:
                        prompt = game.get_game_prompt_round_1(agent_id, agent, turn)
                    else:
                        prompt = game.get_game_prompt_later_round(agent_id, turn, sim_data, last_responses)
                    
                    response = agent.respond(prompt)
                    agent_responses[agent_id] = response
                    
                    print(f"\n{agent_id} ({role_name}) prompt: {prompt}")
                    print(f"{agent_id} ({role_name}) response: {response}")
                    
                    #Allow game to update state after each move (for sequential games)
                    if hasattr(game, 'process_intermediate_response'):
                        game.process_intermediate_response(agent_id, response, turn, sim_data)
                
                # Process turn with game logic
                last_responses = game.process_turn(turn, agent_responses, sim_data)
                
            elif task == "myth":
                # PHASE 2: MYTH WRITING 
                print("\n--- PHASE 2: MYTH WRITING ---")
                
                # Sequential MYTH writing
                # # Use same order as game moves for consistency
                # move_order = game.get_move_order(turn, sim_data)
                # for agent_id in move_order:
                #     agent = sim_data.agents[agent_id]
                #     current_role = "Investor" if agent_id == roles["investor"] else "Trustee"
                    
                #     # Get the myth prompt and response
                #     if turn == 1:
                #         myth_prompt = myth_writer.get_myth_prompt_round_1(agent_id, turn, sim_data)
                #     else:
                #         myth_prompt = myth_writer.get_myth_prompt_round_later(agent_id, turn, sim_data)
                #     myth_response = agent.respond(myth_prompt)
                #     agent_myths[agent_id] = myth_response
                    
                #     print(f"\n{agent_id} ({current_role}) myth prompt:\n{myth_prompt}")
                #     print(f"\n{agent_id} ({current_role}) myth response:\n{myth_response}")

                # myth_writer.process_myths(turn, agent_myths, sim_data)
        
                # PARALLELIZED MYTH WRITING
                # Prepare prompts for all agents (no dependencies)
                prompts = {}
                for agent_id in move_order:
                    if turn == 1:
                        prompts[agent_id] = myth_writer.get_myth_prompt_round_1(agent_id, turn, sim_data)
                    else:
                        prompts[agent_id] = myth_writer.get_myth_prompt_round_later(agent_id, turn, sim_data)

                # Parallelize LLM calls for myth writing
                with ThreadPoolExecutor(max_workers=len(move_order)) as executor:
                    futures = {
                        agent_id: executor.submit(sim_data.agents[agent_id].respond, prompts[agent_id])
                        for agent_id in move_order
                    }
                    
                    # Collect results as they complete
                    for agent_id, future in futures.items():
                        agent_myths[agent_id] = future.result()
                        current_role = "Investor" if agent_id == roles["investor"] else "Trustee"
                        print(f"\n{agent_id} ({current_role}) myth prompt:\n{prompts[agent_id]}")
                        print(f"\n{agent_id} ({current_role}) myth response:\n{agent_myths[agent_id]}")

                myth_writer.process_myths(turn, agent_myths, sim_data)

        # Print turn summary (only if game was run)
        if "game" in task_order:
            game.print_turn_summary(turn, agent_responses, sim_data)
        
        # Print myths (only if myth was run)
        if "myth" in task_order and sim_data.conversation_history and sim_data.conversation_history[-1].get("myths"):
            print(f"\n{'~' * 80}")
            print("MYTHS WRITTEN THIS ROUND:")
            print(f"{'~' * 80}")
            for agent_id, myth in sim_data.conversation_history[-1]["myths"].items():
                current_role = "Investor" if agent_id == roles["investor"] else "Trustee"
                print(f"\n{agent_id} ({current_role}):")
                print(myth)
                print("-" * 40)
    
    # Print game summary (only if game was run)
    if "game" in task_order:
        game.print_game_summary(sim_data)
    print_llm_error_summary()
    return sim_data
