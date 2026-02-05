"""
Run Donor Game simulations with 4 different task configurations.

Configurations:
1. Myth only
2. Game only
3. Myth + Game (myth first, then game)
4. Game + Myth (game first, then myth)

Each configuration runs for 100 rounds with 10 agents using gpt-4o-mini.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.donor_simulation import run_donor_simulation
from src.myth_writer import MythWriter
from games.donor_game import DonorGame
import os
from datetime import datetime


def run_experiment(task_order, output_name, num_turns=100):
    """Run a single experiment configuration."""
    print(f"\n{'#' * 80}")
    print(f"# STARTING EXPERIMENT: {output_name}")
    print(f"# Task order: {task_order}")
    print(f"# Rounds: {num_turns}")
    print(f"{'#' * 80}\n")

    game = DonorGame(
        initial_resources=100,
        donation_multiplier=2,
        population_size=10,
        context_depth=2,
        context_width=2,
    )

    # Only create myth_writer if myths are in task_order
    myth_writer = None
    if "myth" in task_order:
        myth_writer = MythWriter(myth_topic="cooperation and sharing")

    sim_data = run_donor_simulation(
        game=game,
        model="gpt-4o-mini",
        temperature=0.8,
        num_turns=num_turns,
        memory_capacity=5,
        agent_biases=None,
        myth_writer=myth_writer,
        task_order=task_order,
        myth_visibility="all",
        myth_partner_rounds=2,
    )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_path = "data/json/donor_game/"
    os.makedirs(base_path, exist_ok=True)
    output_file = f"{base_path}{output_name}_{timestamp}.json"
    sim_data.save_state(output_file)
    print(f"\nResults saved to {output_file}")

    return sim_data, output_file


if __name__ == "__main__":
    num_turns = 100

    configs = [
        (["myth"], "myth_only"),
        (["game"], "game_only"),
        (["myth", "game"], "myth_then_game"),
        (["game", "myth"], "game_then_myth"),
    ]

    results = []

    for task_order, name in configs:
        sim_data, output_file = run_experiment(task_order, name, num_turns)
        results.append((name, output_file))

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 80)
    print("\nOutput files:")
    for name, filepath in results:
        print(f"  {name}: {filepath}")
