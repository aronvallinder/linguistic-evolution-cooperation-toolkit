"""
Run batch Donor Game experiments from configuration.

Usage:
    python experiments/run_donor_game_batch.py <experiment_name>

Example:
    python experiments/run_donor_game_batch.py donor_pilot
    python experiments/run_donor_game_batch.py donor_with_myths

Supported LLM providers (auto-detected from environment variables):
  - OpenAI:    export OPENAI_API_KEY="sk-..."
  - Gemini:    export GEMINI_API_KEY="..."
  - OpenRouter: export OPENROUTER_API_KEY="sk-or-..."

To force a specific provider: export LLM_PROVIDER="openai" (or "gemini", "openrouter")
"""

import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import yaml
from src.donor_simulation import run_donor_simulation
from src.myth_writer import MythWriter
from src.utils import get_llm_provider, get_default_model
from games.donor_game import DonorGame


def load_config(config_path='config/experiments.yaml'):
    """Load experiment configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_donor_experiment_combinations(config, experiment_name):
    """
    Get all combinations for a donor game experiment set.

    Returns a list of dicts with all parameter combinations to run.
    """
    if experiment_name not in config.get('donor_experiment_sets', {}):
        raise ValueError(f"Donor experiment set '{experiment_name}' not found in config")

    exp_set = config['donor_experiment_sets'][experiment_name]
    base_models = config.get('base_models', {})
    personas = config.get('personas', {})

    # Detect provider and get default model if needed
    provider = get_llm_provider()
    default_model = get_default_model(provider)

    combinations = []

    # Expand models
    models = exp_set.get('models', ['default'])
    if models == 'all':
        models = list(base_models.keys()) if base_models else ['default']

    # Expand personas
    persona_names = exp_set.get('personas', ['neutral'])
    if persona_names == 'all':
        persona_names = list(personas.keys())

    # Get task orders
    task_orders = exp_set.get('task_orders', [['game']])

    # Get base game params
    base_game_params = exp_set.get('game_params', {})

    # Handle special expansion options
    myth_visibility_options = exp_set.get('myth_visibility_options', [base_game_params.get('myth_visibility', 'all')])
    context_variations = exp_set.get('context_variations', None)

    # Generate all combinations
    for model_key in models:
        # Use default model for current provider if 'default' or not found in base_models
        if model_key == 'default':
            model = default_model
        else:
            model = base_models.get(model_key, default_model)

        for persona_name in persona_names:
            persona = personas.get(persona_name, {'description': persona_name, 'system_addition': ''})

            for task_order in task_orders:
                # Handle context variations if specified
                if context_variations:
                    for ctx_var in context_variations:
                        game_params = {**base_game_params, **ctx_var}
                        for myth_vis in myth_visibility_options:
                            game_params_final = {**game_params, 'myth_visibility': myth_vis}
                            combinations.append({
                                'model': model,
                                'model_key': model_key,
                                'persona': persona,
                                'persona_name': persona_name,
                                'task_order': task_order,
                                'game_params': game_params_final,
                            })
                else:
                    for myth_vis in myth_visibility_options:
                        game_params = {**base_game_params, 'myth_visibility': myth_vis}
                        combinations.append({
                            'model': model,
                            'model_key': model_key,
                            'persona': persona,
                            'persona_name': persona_name,
                            'task_order': task_order,
                            'game_params': game_params,
                        })

    return combinations


def run_donor_experiment_set(experiment_name: str):
    """Run all combinations for a donor game experiment set."""
    config = load_config('config/experiments.yaml')
    combinations = get_donor_experiment_combinations(config, experiment_name)

    print(f"Running donor experiment '{experiment_name}' with {len(combinations)} combinations")

    for i, combo in enumerate(combinations):
        print(f"\n{'=' * 80}")
        print(f"--- Combination {i+1}/{len(combinations)} ---")
        print(f"Model: {combo['model_key']}")
        print(f"Persona: {combo['persona_name']}")
        print(f"Task Order: {combo['task_order']}")
        print(f"Game Params: {combo['game_params']}")
        print(f"{'=' * 80}")

        game_params = combo['game_params']

        # Prepare personas dict for all agents
        population_size = game_params.get('population_size', 6)
        personas = {
            f'Agent_{i+1}': combo['persona']
            for i in range(population_size)
        }

        # Create game
        game = DonorGame(
            initial_resources=game_params.get('initial_resources', 100),
            donation_multiplier=game_params.get('donation_multiplier', 2),
            population_size=population_size,
            context_depth=game_params.get('context_depth', 2),
            context_width=game_params.get('context_width', 2),
            personas=personas,
        )

        # Create myth writer if myths are in task order
        myth_writer = None
        if 'myth' in combo['task_order']:
            myth_writer = MythWriter(myth_topic="cooperation and generosity")

        # Run simulation
        sim_data = run_donor_simulation(
            game=game,
            model=combo['model'],
            temperature=game_params.get('temperature', 0.8),
            num_turns=game_params.get('num_turns', 10),
            memory_capacity=game_params.get('memory_capacity', 5),
            agent_biases=None,
            myth_writer=myth_writer,
            task_order=combo['task_order'],
            myth_visibility=game_params.get('myth_visibility', 'all'),
            myth_partner_rounds=game_params.get('myth_partner_rounds', 2),
        )

        # Generate descriptive filename
        model_name = combo['model_key']
        task_order_str = "_".join(combo['task_order'])
        depth = game_params.get('context_depth', 2)
        width = game_params.get('context_width', 2)
        myth_vis = game_params.get('myth_visibility', 'all')

        filename = (
            f"{experiment_name}_{i:03d}_{model_name}_{combo['persona_name']}_"
            f"{task_order_str}_d{depth}w{width}_{myth_vis}.json"
        )
        save_path = f"data/json/donor_game/{experiment_name}/{filename}"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        sim_data.save_state(save_path)
        print(f"Saved to {save_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        experiment_name = sys.argv[1]
        run_donor_experiment_set(experiment_name)
    else:
        print("Usage: python experiments/run_donor_game_batch.py <experiment_name>")
        print("\nAvailable donor experiments:")
        config = load_config('config/experiments.yaml')
        for exp_name in config.get('donor_experiment_sets', {}).keys():
            print(f"  - {exp_name}")
