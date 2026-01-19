"""
Myth Similarity Embedding Analysis for N-Agent Simulations

This script analyzes semantic similarity between myths using sentence embeddings.
Supports arbitrary number of agents (adapted from 2-agent version).

Features:
- Similarity to first round (myth drift over time)
- Cross-agent similarity matrix (population convergence)
- Pairwise similarity trajectories
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations

# Global model variable (loaded once)
model = None


def load_model(model_name='all-mpnet-base-v2'):
    """Load the embedding model."""
    global model
    if model is None:
        print("Loading model...")
        model = SentenceTransformer(model_name)
        print("Model loaded!\n")
    return model


def extract_myths_by_agent(data):
    """
    Extract myths organized by agent.

    Returns:
        Dictionary mapping agent_id -> [(round, myth), ...]
    """
    agent_myths = {}

    for entry in data.get('conversation_history', []):
        round_num = entry.get('round')
        myths = entry.get('myths', {})

        for agent_id, myth in myths.items():
            if agent_id not in agent_myths:
                agent_myths[agent_id] = []
            agent_myths[agent_id].append((round_num, myth))

    # Sort by round
    for agent_id in agent_myths:
        agent_myths[agent_id].sort(key=lambda x: x[0])

    return agent_myths


def analyze_similarity_to_first(agent_myths, embedding_model):
    """
    Analyze how each agent's myths drift from their first myth.

    Returns:
        Dictionary mapping agent_id -> {'rounds': [], 'scores': []}
    """
    similarity_data = {}

    print("=" * 80)
    print("SIMILARITY TO FIRST ROUND (Myth Drift)")
    print("=" * 80)

    for agent_id in sorted(agent_myths.keys()):
        myths = agent_myths[agent_id]
        if len(myths) < 2:
            continue

        first_round, first_myth = myths[0]
        first_emb = embedding_model.encode([first_myth])[0]

        similarity_data[agent_id] = {'rounds': [], 'scores': []}

        print(f"\n{agent_id}:")
        print("-" * 40)

        for current_round, current_myth in myths[1:]:
            current_emb = embedding_model.encode([current_myth])[0]
            sim = cosine_similarity(first_emb.reshape(1, -1), current_emb.reshape(1, -1))[0][0]

            similarity_data[agent_id]['rounds'].append(current_round)
            similarity_data[agent_id]['scores'].append(sim)

            print(f"  Round {first_round} → {current_round}: {sim*10:.2f}/10 (cosine: {sim:.4f})")

    return similarity_data


def analyze_cross_agent_similarity(agent_myths, embedding_model):
    """
    Analyze similarity between agents' myths at each round.

    Returns:
        Dictionary with:
        - 'rounds': list of round numbers
        - 'pairwise': {(agent_a, agent_b): [similarities per round]}
        - 'population_mean': [mean similarity per round]
    """
    agents = sorted(agent_myths.keys())

    # Get all rounds (assuming all agents have same rounds)
    rounds = [r for r, _ in agent_myths[agents[0]]]

    # Build round -> agent -> myth mapping
    myths_by_round = {}
    for agent_id, myth_list in agent_myths.items():
        for round_num, myth in myth_list:
            if round_num not in myths_by_round:
                myths_by_round[round_num] = {}
            myths_by_round[round_num][agent_id] = myth

    # Calculate pairwise similarities per round
    pairwise_sims = {pair: [] for pair in combinations(agents, 2)}
    population_means = []

    print("\n" + "=" * 80)
    print("CROSS-AGENT SIMILARITY (Population Convergence)")
    print("=" * 80)

    for round_num in rounds:
        round_myths = myths_by_round.get(round_num, {})
        if len(round_myths) < 2:
            continue

        # Encode all myths for this round
        agent_list = sorted(round_myths.keys())
        embeddings = {a: embedding_model.encode([round_myths[a]])[0] for a in agent_list}

        # Calculate all pairwise similarities
        round_sims = []
        for agent_a, agent_b in combinations(agent_list, 2):
            sim = cosine_similarity(
                embeddings[agent_a].reshape(1, -1),
                embeddings[agent_b].reshape(1, -1)
            )[0][0]

            if (agent_a, agent_b) in pairwise_sims:
                pairwise_sims[(agent_a, agent_b)].append(sim)
            round_sims.append(sim)

        mean_sim = np.mean(round_sims) if round_sims else 0
        population_means.append(mean_sim)

        print(f"\nRound {round_num}: Mean similarity = {mean_sim*10:.2f}/10")

    return {
        'rounds': rounds,
        'pairwise': pairwise_sims,
        'population_mean': population_means,
        'agents': agents,
    }


def analyze_final_similarity_matrix(agent_myths, embedding_model):
    """
    Create a similarity matrix for the final round.

    Returns:
        (agents, similarity_matrix)
    """
    agents = sorted(agent_myths.keys())
    n_agents = len(agents)

    # Get final myths
    final_myths = {}
    for agent_id in agents:
        if agent_myths[agent_id]:
            _, myth = agent_myths[agent_id][-1]
            final_myths[agent_id] = myth

    # Encode all final myths
    embeddings = {a: embedding_model.encode([final_myths[a]])[0] for a in agents}

    # Build similarity matrix
    sim_matrix = np.zeros((n_agents, n_agents))
    for i, agent_a in enumerate(agents):
        for j, agent_b in enumerate(agents):
            if i == j:
                sim_matrix[i, j] = 1.0
            else:
                sim = cosine_similarity(
                    embeddings[agent_a].reshape(1, -1),
                    embeddings[agent_b].reshape(1, -1)
                )[0][0]
                sim_matrix[i, j] = sim

    return agents, sim_matrix


def create_plots(similarity_to_first, cross_agent_data, final_matrix, agents, output_dir):
    """Create and save all visualization plots."""
    print("\n" + "=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)

    n_agents = len(agents)
    colors = plt.cm.tab10(np.linspace(0, 1, min(n_agents, 10)))
    agent_colors = {agent: colors[i % 10] for i, agent in enumerate(agents)}

    # Plot 1: Similarity to first round (all agents)
    fig, ax = plt.subplots(figsize=(12, 6))
    for agent_id in agents:
        if agent_id in similarity_to_first:
            rounds = similarity_to_first[agent_id]['rounds']
            scores = [s * 10 for s in similarity_to_first[agent_id]['scores']]
            ax.plot(rounds, scores, marker='o', linewidth=2, markersize=6,
                   label=agent_id, color=agent_colors[agent_id], alpha=0.8)

    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Similarity to Round 1 (0-10)', fontsize=12)
    ax.set_title('Myth Drift: Similarity to First Round Over Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 10)
    ax.legend(loc='best', fontsize=9, ncol=2 if n_agents > 5 else 1)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/similarity_to_first.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/similarity_to_first.png")
    plt.close()

    # Plot 2: Population mean similarity over time
    fig, ax = plt.subplots(figsize=(10, 6))
    rounds = cross_agent_data['rounds']
    means = [s * 10 for s in cross_agent_data['population_mean']]
    ax.plot(rounds, means, marker='s', linewidth=2.5, markersize=8, color='#2c3e50')
    ax.fill_between(rounds, 0, means, alpha=0.3, color='#2c3e50')

    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Mean Pairwise Similarity (0-10)', fontsize=12)
    ax.set_title('Population Convergence: Mean Cross-Agent Similarity', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 10)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/population_convergence.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/population_convergence.png")
    plt.close()

    # Plot 3: Final round similarity matrix (heatmap)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(final_matrix * 10, cmap='YlOrRd', aspect='auto', vmin=0, vmax=10)

    ax.set_xticks(np.arange(n_agents))
    ax.set_yticks(np.arange(n_agents))
    ax.set_xticklabels(agents, rotation=45, ha='right')
    ax.set_yticklabels(agents)

    ax.set_xlabel('Agent', fontsize=12)
    ax.set_ylabel('Agent', fontsize=12)
    ax.set_title('Final Round: Cross-Agent Myth Similarity Matrix', fontsize=14, fontweight='bold')

    # Add text annotations (only if not too many agents)
    if n_agents <= 12:
        for i in range(n_agents):
            for j in range(n_agents):
                text_color = 'white' if final_matrix[i, j] > 0.7 else 'black'
                ax.text(j, i, f'{final_matrix[i, j]*10:.1f}',
                       ha="center", va="center", color=text_color, fontsize=8)

    plt.colorbar(im, ax=ax, label='Similarity Score (0-10)')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/final_similarity_matrix.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/final_similarity_matrix.png")
    plt.close()

    # Plot 4: Similarity drift heatmap (agents x rounds)
    if similarity_to_first:
        # Build heatmap data
        rounds = similarity_to_first[agents[0]]['rounds'] if agents[0] in similarity_to_first else []
        if rounds:
            heatmap_data = np.zeros((n_agents, len(rounds)))
            for idx, agent_id in enumerate(agents):
                if agent_id in similarity_to_first:
                    scores = similarity_to_first[agent_id]['scores']
                    heatmap_data[idx, :len(scores)] = scores

            fig, ax = plt.subplots(figsize=(12, max(6, n_agents * 0.5)))
            im = ax.imshow(heatmap_data * 10, cmap='YlOrRd', aspect='auto', vmin=0, vmax=10)

            ax.set_xticks(np.arange(len(rounds)))
            ax.set_yticks(np.arange(n_agents))
            ax.set_xticklabels(rounds)
            ax.set_yticklabels(agents)

            ax.set_xlabel('Round', fontsize=12)
            ax.set_ylabel('Agent', fontsize=12)
            ax.set_title('Myth Drift Heatmap: Similarity to Round 1', fontsize=14, fontweight='bold')

            plt.colorbar(im, ax=ax, label='Similarity Score (0-10)')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/drift_heatmap.png', dpi=300, bbox_inches='tight')
            print(f"Saved: {output_dir}/drift_heatmap.png")
            plt.close()

    print("\n" + "=" * 80)
    print("ALL PLOTS GENERATED SUCCESSFULLY")
    print("=" * 80)


def print_summary(similarity_to_first, cross_agent_data, final_matrix, agents):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    # Drift summary
    print("\nMYTH DRIFT (similarity to first round):")
    for agent_id in agents:
        if agent_id in similarity_to_first and similarity_to_first[agent_id]['scores']:
            scores = similarity_to_first[agent_id]['scores']
            print(f"  {agent_id}: start={scores[0]*10:.1f}/10, end={scores[-1]*10:.1f}/10, "
                  f"mean={np.mean(scores)*10:.1f}/10")

    # Convergence summary
    print("\nPOPULATION CONVERGENCE:")
    means = cross_agent_data['population_mean']
    if means:
        print(f"  First round mean similarity: {means[0]*10:.1f}/10")
        print(f"  Final round mean similarity: {means[-1]*10:.1f}/10")
        print(f"  Change: {(means[-1] - means[0])*10:+.1f}")

    # Final matrix summary
    print("\nFINAL ROUND SIMILARITY:")
    upper_triangle = final_matrix[np.triu_indices(len(agents), k=1)]
    print(f"  Mean pairwise similarity: {np.mean(upper_triangle)*10:.1f}/10")
    print(f"  Min: {np.min(upper_triangle)*10:.1f}/10")
    print(f"  Max: {np.max(upper_triangle)*10:.1f}/10")
    print(f"  Std: {np.std(upper_triangle)*10:.1f}")


def main():
    """Main execution function."""
    # Load data
    print(f"Loading simulation data from {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Extract myths
    agent_myths = extract_myths_by_agent(data)

    if not agent_myths:
        print("No myths found in the data. Exiting.")
        return

    print(f"Found {len(agent_myths)} agents with myths")
    for agent_id, myths in sorted(agent_myths.items()):
        print(f"  {agent_id}: {len(myths)} rounds")

    # Load model
    embedding_model = load_model(model_name)

    # Run analyses
    similarity_to_first = analyze_similarity_to_first(agent_myths, embedding_model)
    cross_agent_data = analyze_cross_agent_similarity(agent_myths, embedding_model)
    agents, final_matrix = analyze_final_similarity_matrix(agent_myths, embedding_model)

    # Print summary
    print_summary(similarity_to_first, cross_agent_data, final_matrix, agents)

    # Create plots
    create_plots(similarity_to_first, cross_agent_data, final_matrix, agents, output_dir)

    # Save results to JSON (convert numpy types to native Python)
    results = {
        'similarity_to_first': {
            agent_id: {
                'rounds': data['rounds'],
                'scores': [float(s) for s in data['scores']]
            }
            for agent_id, data in similarity_to_first.items()
        },
        'population_convergence': {
            'rounds': cross_agent_data['rounds'],
            'mean_similarity': [float(s) for s in cross_agent_data['population_mean']],
        },
        'final_similarity_matrix': {
            'agents': agents,
            'matrix': [[float(x) for x in row] for row in final_matrix.tolist()],
        }
    }

    results_file = f'{output_dir}/similarity_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == '__main__':
    # Configuration from environment or defaults
    project_root = Path(__file__).resolve().parents[2]
    default_input = project_root / "data/json/donor_game/simulation_state.json"
    default_output = project_root / "data/plots/_local/donor_myth_similarity"

    input_file = os.environ.get('ANALYSIS_INPUT_FILE', str(default_input))
    base_output_dir = os.environ.get('ANALYSIS_OUTPUT_DIR', str(default_output))
    os.makedirs(base_output_dir, exist_ok=True)

    output_dir = f'{base_output_dir}/similarity_embedding'
    model_name = 'all-mpnet-base-v2'

    main()
