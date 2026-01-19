"""
Run a single Donor Game simulation.

This script runs one simulation of the Donor Game with specified parameters.
For batch experiments with multiple configurations, use run_donor_game_batch.py.

Supported LLM providers (auto-detected from environment variables):
  - OpenAI:    export OPENAI_API_KEY="sk-..."
  - Gemini:    export GEMINI_API_KEY="..."
  - OpenRouter: export OPENROUTER_API_KEY="sk-or-..."

To force a specific provider: export LLM_PROVIDER="openai" (or "gemini", "openrouter")
"""

from src.donor_simulation import run_donor_simulation
from src.myth_writer import MythWriter
from src.utils import get_default_model
from games.donor_game import DonorGame
import os


if __name__ == "__main__":
    # Configure game parameters
    game = DonorGame(
        initial_resources=100,
        donation_multiplier=2,
        population_size=6,  # Must be even
        context_depth=2,    # How far back in donation chains to trace
        context_width=2,    # How many parallel branches to show
    )

    # Configure myth writer (optional - set to None to disable myths)
    myth_writer = MythWriter(myth_topic="cooperation and sharing")

    # LLM configuration - auto-detect provider and use appropriate default model
    # Or override with a specific model name for your provider:
    #   OpenAI models: "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"
    #   Gemini models: "gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"
    #   OpenRouter: "meta-llama/llama-3.1-8b-instruct", "anthropic/claude-3-sonnet", etc.
    model = get_default_model()  # Auto-selects based on provider

    # Run simulation
    sim_data = run_donor_simulation(
        game=game,
        model=model,
        temperature=0.8,
        num_turns=10,
        memory_capacity=5,
        agent_biases=None,  # Can pass list of biases per agent
        myth_writer=myth_writer,
        task_order=["game", "myth"],  # Options: ["game"], ["game", "myth"], ["myth", "game"]
        myth_visibility="all",  # Options: "all", "recent_partners"
        myth_partner_rounds=2,  # Only used if myth_visibility="recent_partners"
    )

    # Save results
    base_path = "data/json/donor_game/"
    os.makedirs(base_path, exist_ok=True)
    output_file = base_path + "donor_game_test.json"
    sim_data.save_state(output_file)
    print(f"\nSimulation state saved to {output_file}")
