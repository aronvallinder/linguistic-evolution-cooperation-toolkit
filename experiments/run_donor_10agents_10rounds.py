"""
Run Donor Game: 10 agents, 10 rounds.

Usage:
    export OPENAI_API_KEY="sk-..."
    python experiments/run_donor_10agents_10rounds.py
"""

import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.donor_simulation import run_donor_simulation
from src.myth_writer import MythWriter
from src.utils import get_default_model, get_llm_provider
from games.donor_game import DonorGame


if __name__ == "__main__":
    # Verify API key is set
    provider = get_llm_provider()
    model = get_default_model()
    print(f"Using provider: {provider}")
    print(f"Using model: {model}")

    # Configure game: 10 agents, 10 rounds
    game = DonorGame(
        initial_resources=100,
        donation_multiplier=2,
        population_size=10,  # 10 agents
        context_depth=2,
        context_width=2,
    )

    # Enable myth writing
    myth_writer = MythWriter(myth_topic="cooperation and sharing")

    # Run simulation: myths first, then game
    sim_data = run_donor_simulation(
        game=game,
        model=model,
        temperature=0.8,
        num_turns=10,  # 10 rounds
        memory_capacity=5,
        agent_biases=None,
        myth_writer=myth_writer,
        task_order=["myth", "game"],  # Myths first, then game decisions
        myth_visibility="all",
    )

    # Save results
    base_path = "data/json/donor_game/"
    os.makedirs(base_path, exist_ok=True)
    output_file = base_path + "donor_10agents_10rounds.json"
    sim_data.save_state(output_file)
    print(f"\nSimulation state saved to {output_file}")
