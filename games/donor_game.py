from games.base_game import Game
import random
import re


class DonorGame(Game):
    """
    Donor Game: A population-based economic game where agents can transfer
    resources to partners who receive a multiplied benefit.

    Key features:
    - Population of N agents (must be even for pairing)
    - Random bidirectional pairing each round
    - Simultaneous decisions within pairs
    - Two-step decision process: reasoning, then donation amount
    - Observable history tree with configurable depth/width
    """

    def __init__(
        self,
        initial_resources,
        donation_multiplier,
        population_size,
        context_depth=1,
        context_width=1,
        system_prompt_template=None,
        personas=None,
    ):
        super().__init__()

        if population_size % 2 != 0:
            raise ValueError("Population size must be even for pairing.")

        self.initial_resources = initial_resources
        self.donation_multiplier = donation_multiplier
        self.population_size = population_size
        self.context_depth = context_depth
        self.context_width = context_width
        self.system_prompt_template = system_prompt_template
        self.personas = personas or {}

        # Will be set when simulation starts
        self.sim_data_ref = None

    def get_agent_ids(self):
        """Return list of all agent IDs."""
        return [f"Agent_{i+1}" for i in range(self.population_size)]

    def get_system_prompt(self, agent_id, agent):
        """Return the system prompt describing game rules."""
        if self.system_prompt_template:
            base_prompt = self.system_prompt_template.format(
                initial_resources=self.initial_resources,
                donation_multiplier=self.donation_multiplier,
                population_size=self.population_size,
            )
        else:
            base_prompt = f"""You are participating in a Donor Game with {self.population_size} agents.

GAME RULES:
- Each agent starts with {self.initial_resources} resource units
- Each round, you receive a fresh allocation of {self.initial_resources} units
- You decide how much of that round's allocation to donate to your partner, and you keep the rest
- Donations are multiplied: if you donate X, your partner receives X × {self.donation_multiplier}
- Your partner also decides how much to donate to you (simultaneously)
- You can donate any amount from 0 up to {self.initial_resources} (your round allocation)
- Resources accumulate across rounds, but each round's donation decision is over the fixed allocation of {self.initial_resources}

DECISION PROCESS:
Each round you will be asked to:
1. First, provide your reasoning about how much to donate
2. Then, specify the exact donation amount

Your goal is to navigate this social dilemma effectively."""

        # Add persona if specified for this agent
        if agent_id in self.personas and self.personas[agent_id].get('system_addition'):
            base_prompt += f"\n\n{self.personas[agent_id]['system_addition']}"

        return base_prompt

    def initialize_game_data(self, sim_data):
        """Initialize game state for a new simulation."""
        sim_data.game_data = {
            "balances": {
                agent_id: self.initial_resources
                for agent_id in self.get_agent_ids()
            },
            "donation_history": [],  # List of all donations across rounds
            "pairings_history": [],  # List of pairings per round
        }
        self.sim_data_ref = sim_data

    def generate_pairings(self, turn, sim_data):
        """
        Generate random bidirectional pairings for this round.
        Returns list of (agent_a, agent_b) tuples where both donate to each other.
        """
        agents = list(sim_data.agents.keys())
        random.shuffle(agents)

        pairings = []
        for i in range(0, len(agents), 2):
            pairings.append((agents[i], agents[i + 1]))

        return pairings

    def get_roles_for_round(self, turn):
        """
        For Donor Game, there are no fixed roles like investor/trustee.
        All agents are donors to their paired partner.
        Returns empty dict as roles aren't used the same way.
        """
        return {}

    def get_move_order(self, turn, sim_data):
        """
        For Donor Game with simultaneous moves, return all agents.
        The actual move order doesn't matter since decisions are independent.
        """
        return list(sim_data.agents.keys())

    def get_observable_history(self, observer_id, partner_id, sim_data):
        """
        Build the observable history tree for observer looking at partner.

        Parameters:
        - observer_id: The agent making the decision
        - partner_id: The agent they're paired with
        - sim_data: Simulation data containing donation history

        Returns a formatted string describing observable donation history.
        """
        donation_history = sim_data.game_data.get("donation_history", [])

        if not donation_history:
            return "No previous donation history available."

        # Build history tree using BFS with depth/width limits
        history_lines = []

        # Start with partner's donations (depth 0 from partner's perspective)
        self._build_history_tree(
            agent_id=partner_id,
            donation_history=donation_history,
            current_depth=0,
            max_depth=self.context_depth,
            width=self.context_width,
            history_lines=history_lines,
            indent=0,
            visited_chains=set(),
        )

        if not history_lines:
            return "No observable donation history for this partner."

        return "\n".join(history_lines)

    def _build_history_tree(
        self,
        agent_id,
        donation_history,
        current_depth,
        max_depth,
        width,
        history_lines,
        indent,
        visited_chains,
    ):
        """
        Recursively build the observable history tree.

        Shows donations made BY agent_id, then traces to recipients' behavior.
        """
        if current_depth > max_depth:
            return

        # Find donations made by this agent, sorted by round (most recent first)
        agent_donations = [
            d for d in donation_history
            if d["donor_id"] == agent_id
        ]
        agent_donations.sort(key=lambda x: x["round"], reverse=True)

        # Limit by width
        agent_donations = agent_donations[:width]

        indent_str = "  " * indent

        for donation in agent_donations:
            # Create a chain identifier to avoid loops
            chain_id = (donation["donor_id"], donation["recipient_id"], donation["round"])
            if chain_id in visited_chains:
                continue
            visited_chains.add(chain_id)

            round_num = donation["round"]
            recipient_id = donation["recipient_id"]
            percentage = donation["percentage"]

            history_lines.append(
                f"{indent_str}Round {round_num}: {agent_id} donated {percentage:.1f}% of their resources to {recipient_id}"
            )

            # Recurse: show what the recipient did in subsequent rounds
            if current_depth < max_depth:
                self._build_history_tree(
                    agent_id=recipient_id,
                    donation_history=donation_history,
                    current_depth=current_depth + 1,
                    max_depth=max_depth,
                    width=width,
                    history_lines=history_lines,
                    indent=indent + 1,
                    visited_chains=visited_chains,
                )

    def get_reasoning_prompt(self, agent_id, partner_id, turn, sim_data):
        """
        Generate the first prompt asking for reasoning about the donation decision.
        """
        balance = sim_data.game_data["balances"][agent_id]
        history = self.get_observable_history(agent_id, partner_id, sim_data)

        prompt = f"""ROUND {turn}

CURRENT SITUATION:
- Your total accumulated resources: {balance} units
- This round's allocation: {self.initial_resources} units (you decide how to split this between yourself and your partner)
- You are paired with: {partner_id}
- Donation multiplier: {self.donation_multiplier}x (if you donate X, your partner receives {self.donation_multiplier}X)
- Total rounds in game: This information will be revealed as the game progresses

OBSERVABLE HISTORY OF YOUR PARTNER:
{history}

Please provide your reasoning about how much you want to donate to {partner_id} this round.
Consider:
- What you can observe about your partner's past behavior
- The potential benefits and risks of different donation amounts

Provide your reasoning now:"""

        return prompt

    def get_donation_prompt(self, agent_id, partner_id, turn, sim_data):
        """
        Generate the second prompt asking for the specific donation amount.
        """
        allocation = self.initial_resources

        prompt = f"""Based on your reasoning, specify your donation amount.

This round's allocation: {allocation} units
You can donate any amount from 0 to {allocation}. You keep whatever you don't donate.

Respond with exactly this format:
{{"donation": <amount>}}

Where <amount> is a number between 0 and {allocation}."""

        return prompt

    def extract_donation_amount(self, response, max_amount):
        """
        Extract the donation amount from the agent's response.
        Clamps to valid range [0, max_amount].
        """
        # Try to find JSON format first
        pattern = r'"donation"\s*:\s*(\d+\.?\d*)'
        match = re.search(pattern, response, re.IGNORECASE)

        if match:
            amount = float(match.group(1))
        else:
            # Fallback: try to find any number
            numbers = re.findall(r'\d+\.?\d*', response)
            if numbers:
                amount = float(numbers[-1])  # Take the last number
            else:
                # Default to 0 if no number found
                amount = 0.0

        # Clamp to valid range
        amount = max(0, min(amount, max_amount))
        return amount

    def process_turn(self, turn, agent_responses, sim_data):
        """
        Process all donations for this round.

        agent_responses format: {
            agent_id: {
                "reasoning": str,
                "donation_response": str,
                "donation_amount": float,
                "partner_id": str,
            }
        }
        """
        # Initialize if needed
        if "balances" not in sim_data.game_data:
            self.initialize_game_data(sim_data)

        # Each agent receives their round allocation
        allocation = self.initial_resources

        # Collect all donations before applying them (simultaneous)
        donations_this_round = []

        for agent_id, response_data in agent_responses.items():
            partner_id = response_data["partner_id"]
            donation_amount = response_data["donation_amount"]
            donor_balance = sim_data.game_data["balances"][agent_id]

            # Calculate percentage of round allocation donated
            percentage = (donation_amount / allocation * 100) if allocation > 0 else 0

            donations_this_round.append({
                "round": turn,
                "donor_id": agent_id,
                "recipient_id": partner_id,
                "amount": donation_amount,
                "percentage": percentage,
                "donor_balance_before": donor_balance,
            })

        # Apply round allocation and donations simultaneously
        # Each agent gets: +allocation (round income) - donation_given + donations_received * multiplier
        balance_changes = {agent_id: allocation for agent_id in sim_data.agents.keys()}

        for donation in donations_this_round:
            donor_id = donation["donor_id"]
            recipient_id = donation["recipient_id"]
            amount = donation["amount"]

            # Donor loses donated amount from their allocation
            balance_changes[donor_id] -= amount
            # Recipient gains amount * multiplier
            balance_changes[recipient_id] += amount * self.donation_multiplier

        # Update balances
        for agent_id, change in balance_changes.items():
            sim_data.game_data["balances"][agent_id] += change

        # Store donation history
        sim_data.game_data["donation_history"].extend(donations_this_round)

        # Update conversation history entry for this round
        pairings = sim_data.game_data.get("current_pairings", [])

        for entry in sim_data.conversation_history:
            if entry["round"] == turn:
                entry["donations"] = donations_this_round
                entry["pairings"] = pairings
                entry["balances"] = dict(sim_data.game_data["balances"])
                entry["actions"] = {
                    agent_id: {
                        "partner": response_data["partner_id"],
                        "donation": response_data["donation_amount"],
                        "reasoning": response_data["reasoning"],
                    }
                    for agent_id, response_data in agent_responses.items()
                }
                break

        # Store pairings history
        sim_data.game_data["pairings_history"].append({
            "round": turn,
            "pairings": pairings,
        })

        return {
            agent_id: {"donated": response_data["donation_amount"]}
            for agent_id, response_data in agent_responses.items()
        }

    def print_turn_summary(self, turn, agent_responses, sim_data):
        """Print summary of this round's donations."""
        print(f"\n{'*' * 80}")
        print(f"ROUND {turn} COMPLETE")
        print(f"{'*' * 80}")

        # Find the entry for this round
        round_entry = None
        for entry in sim_data.conversation_history:
            if entry["round"] == turn:
                round_entry = entry
                break

        if round_entry and "donations" in round_entry:
            print("\nDonations this round:")
            for donation in round_entry["donations"]:
                print(
                    f"  {donation['donor_id']} → {donation['recipient_id']}: "
                    f"{donation['amount']:.1f} units ({donation['percentage']:.1f}% of resources)"
                )

            print("\nCurrent balances:")
            for agent_id, balance in sorted(round_entry["balances"].items()):
                print(f"  {agent_id}: {balance:.1f} units")

        print(f"{'*' * 80}")

    def print_game_summary(self, sim_data):
        """Print final game summary with statistics."""
        print("\n" + "=" * 80)
        print("DONOR GAME SUMMARY")
        print("=" * 80)

        donation_history = sim_data.game_data.get("donation_history", [])
        balances = sim_data.game_data.get("balances", {})

        # Total rounds played
        rounds_played = max((d["round"] for d in donation_history), default=0)
        print(f"\nTotal rounds played: {rounds_played}")
        print(f"Population size: {self.population_size}")
        print(f"Initial resources per agent: {self.initial_resources}")
        print(f"Donation multiplier: {self.donation_multiplier}x")

        # Overall statistics
        if donation_history:
            total_donated = sum(d["amount"] for d in donation_history)
            avg_donation_pct = sum(d["percentage"] for d in donation_history) / len(donation_history)

            print(f"\nOverall statistics:")
            print(f"  Total resources donated: {total_donated:.1f}")
            print(f"  Average donation percentage: {avg_donation_pct:.1f}%")

        # Final balances
        print("\nFinal balances:")
        sorted_balances = sorted(balances.items(), key=lambda x: x[1], reverse=True)
        for agent_id, balance in sorted_balances:
            change = balance - self.initial_resources
            sign = "+" if change >= 0 else ""
            print(f"  {agent_id}: {balance:.1f} ({sign}{change:.1f} from start)")

        # Per-agent donation statistics
        print("\nPer-agent donation behavior:")
        for agent_id in self.get_agent_ids():
            agent_donations = [d for d in donation_history if d["donor_id"] == agent_id]
            if agent_donations:
                avg_pct = sum(d["percentage"] for d in agent_donations) / len(agent_donations)
                total = sum(d["amount"] for d in agent_donations)
                print(f"  {agent_id}: avg {avg_pct:.1f}% donated, total {total:.1f} given")

        print("=" * 80)
