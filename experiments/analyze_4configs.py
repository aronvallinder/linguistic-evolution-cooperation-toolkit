"""
Analyze all 4 Donor Game experiment configurations.

Runs trajectory plotting, cooperativity analysis, and myth similarity embedding
for each configuration.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import json

# Import analysis modules
from analyses_clean.DonorGame.trajectory_plotting import (
    load_simulation_data,
    calculate_trajectory_metrics,
    plot_numerical_trajectories,
    plot_aggregate_statistics,
    extract_per_agent_data,
)
from analyses_clean.DonorGame.cooperativity_analysis import (
    read_myths_from_json,
    analyze_cooperativity,
    print_cooperativity_summary,
    visualize_cooperativity,
    save_results_to_json as save_cooperativity_json,
)


def analyze_experiment(input_file: str, output_dir: str, name: str):
    """Run all analyses for a single experiment."""
    print(f"\n{'#' * 80}")
    print(f"# ANALYZING: {name}")
    print(f"# Input: {input_file}")
    print(f"# Output: {output_dir}")
    print(f"{'#' * 80}\n")

    os.makedirs(output_dir, exist_ok=True)

    # Load data
    data = load_simulation_data(input_file)
    conversation_history = data.get('conversation_history', [])
    game_data = data.get('game_data', {})
    task_order = data.get('task_order', [])

    print(f"Task order: {task_order}")
    print(f"Rounds: {len(conversation_history)}")

    # =========================================================================
    # TRAJECTORY ANALYSIS (if game was played)
    # =========================================================================
    if 'game' in task_order:
        print("\n--- TRAJECTORY ANALYSIS ---")

        metrics = calculate_trajectory_metrics(conversation_history, game_data)

        print("\nTrajectory Metrics:")
        print(f"  Total Rounds: {metrics['total_rounds']}")
        print(f"  Total Donations: {metrics['total_donations']:.2f}")
        print(f"  Average Donation %: {metrics['avg_donation_percentage']:.2f}%")
        print(f"  Generosity Score: {metrics['generosity_score']:.2f}%")
        print(f"  Final Balance Gini: {metrics['final_balance_gini']:.3f}")

        # Save metrics
        with open(f"{output_dir}/trajectory_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)

        # Generate plots
        plot_numerical_trajectories(
            conversation_history,
            game_data,
            save_path=f"{output_dir}/trajectory_numerical.png",
            title=f"{name} - Numerical Trajectories"
        )

        plot_aggregate_statistics(
            conversation_history,
            game_data,
            save_path=f"{output_dir}/trajectory_aggregate.png",
            title=f"{name} - Aggregate Statistics"
        )
    else:
        print("\n--- SKIPPING TRAJECTORY ANALYSIS (no game data) ---")

    # =========================================================================
    # COOPERATIVITY ANALYSIS (if myths were written)
    # =========================================================================
    if 'myth' in task_order:
        print("\n--- COOPERATIVITY ANALYSIS ---")

        myths_by_agent = read_myths_from_json(input_file)

        if myths_by_agent:
            print(f"Found {len(myths_by_agent)} agents with myths")

            cooperativity_scores = analyze_cooperativity(myths_by_agent)
            print_cooperativity_summary(cooperativity_scores)

            save_cooperativity_json(
                cooperativity_scores,
                f"{output_dir}/cooperativity_results.json"
            )

            visualize_cooperativity(
                cooperativity_scores,
                output_file=f"{output_dir}/cooperativity_analysis.png"
            )
        else:
            print("No myths found in data")
    else:
        print("\n--- SKIPPING COOPERATIVITY ANALYSIS (no myth data) ---")

    print(f"\n{'=' * 80}")
    print(f"Analysis complete for {name}")
    print(f"Results saved to {output_dir}")
    print(f"{'=' * 80}")


def run_myth_similarity(input_file: str, output_dir: str, name: str):
    """
    Run myth similarity embedding analysis.
    Separated because it requires sentence-transformers which may not be installed.
    """
    try:
        from analyses_clean.DonorGame.myth_similarity_embedding import (
            load_model,
            extract_myths_by_agent,
            analyze_similarity_to_first,
            analyze_cross_agent_similarity,
            analyze_final_similarity_matrix,
            create_plots,
            print_summary,
        )

        print(f"\n--- MYTH SIMILARITY EMBEDDING ANALYSIS: {name} ---")

        with open(input_file, 'r') as f:
            data = json.load(f)

        task_order = data.get('task_order', [])
        if 'myth' not in task_order:
            print("Skipping (no myth data)")
            return

        agent_myths = extract_myths_by_agent(data)
        if not agent_myths:
            print("No myths found")
            return

        print(f"Found {len(agent_myths)} agents with myths")

        embedding_model = load_model()

        similarity_to_first = analyze_similarity_to_first(agent_myths, embedding_model)
        cross_agent_data = analyze_cross_agent_similarity(agent_myths, embedding_model)
        agents, final_matrix = analyze_final_similarity_matrix(agent_myths, embedding_model)

        print_summary(similarity_to_first, cross_agent_data, final_matrix, agents)

        similarity_output_dir = f"{output_dir}/similarity_embedding"
        create_plots(similarity_to_first, cross_agent_data, final_matrix, agents, similarity_output_dir)

        print(f"Myth similarity analysis complete for {name}")

    except ImportError as e:
        print(f"Skipping myth similarity analysis: {e}")
        print("Install sentence-transformers to enable: pip install sentence-transformers")


if __name__ == "__main__":
    # Define experiment files
    experiments = {
        "myth_only": "data/json/donor_game/myth_only_20260120_103146.json",
        "game_only": "data/json/donor_game/game_only_20260120_105048.json",
        "myth_then_game": "data/json/donor_game/myth_then_game_20260120_112411.json",
        "game_then_myth": "data/json/donor_game/game_then_myth_20260120_115741.json",
    }

    base_output_dir = "data/plots/donor_game_4configs"

    # Run basic analyses for all experiments
    for name, input_file in experiments.items():
        output_dir = f"{base_output_dir}/{name}"
        analyze_experiment(input_file, output_dir, name)

    # Run myth similarity analysis (separate loop to load model once)
    print("\n" + "=" * 80)
    print("RUNNING MYTH SIMILARITY EMBEDDING ANALYSIS")
    print("=" * 80)

    for name, input_file in experiments.items():
        output_dir = f"{base_output_dir}/{name}"
        run_myth_similarity(input_file, output_dir, name)

    # Print final summary
    print("\n" + "#" * 80)
    print("# ALL ANALYSES COMPLETE")
    print("#" * 80)
    print("\nOutput directories:")
    for name in experiments.keys():
        print(f"  {name}: {base_output_dir}/{name}/")
