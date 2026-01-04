import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.experiment_config import ExperimentConfig
from src.simulation import run_simulation
from src.myth_writer import MythWriter
from games.trust_game import TrustGame

def run_experiment_set(experiment_name: str):
    # Load configuration
    config = ExperimentConfig('config/experiments.yaml')
    combinations = config.get_experiment_combinations(experiment_name)

    print(f"Running {experiment_name} with {len(combinations)} combinations")

    for i, combo in enumerate(combinations):
        print(f"\n--- Combination {i+1}/{len(combinations)} ---")
        print(f"Model: {combo['model']}")
        print(f"Persona: {combo['persona']['description']}")
        print(f"Task Order: {combo['task_order']}")

        # Configure game
        game_params = combo['game_params']

        # Prepare personas dict for both agents
        personas = {
            'Agent_1': combo['persona'],
            'Agent_2': combo['persona']  # Both agents use same persona for now
        }

        game = TrustGame(
            endowment=game_params['endowment'],
            multiplier=game_params['multiplier'],
            system_prompt_template=combo['template'],
            personas=personas
        )

        myth_writer = MythWriter(myth_topic="")

        # Run simulation
        sim_data = run_simulation(
            game=game,
            model=combo['model'],
            temperature=game_params.get('temperature', 0.8),
            num_turns=game_params['num_turns'],
            num_agents=game_params['num_agents'],
            memory_capacity=game_params['memory_capacity'],
            agent_biases="",
            myth_writer=myth_writer,
            task_order=combo['task_order']
        )

        # Save with descriptive filename
        model_name = combo['model'].split('/')[-1] if '/' in combo['model'] else combo['model']
        task_order_str = "_".join(combo['task_order'])
        filename = f"{experiment_name}_{i:03d}_{model_name}_{combo['persona']['description']}_{task_order_str}.json"
        save_path = f"data/json/{experiment_name}/{filename}"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        sim_data.save_state(save_path)
        print(f"✓ Saved to {save_path}")

if __name__ == "__main__":
    # Accept experiment set name as command line argument
    if len(sys.argv) > 1:
        experiment_name = sys.argv[1]
        run_experiment_set(experiment_name)
    else:
        # Run default experiment sets
        run_experiment_set("pilot")
        run_experiment_set("persona_comparison")
        # run_experiment_set("full_factorial")  # Comment out for now