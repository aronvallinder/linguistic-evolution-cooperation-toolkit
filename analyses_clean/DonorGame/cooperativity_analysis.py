"""
Cooperativity Analysis for Donor Game Myths

This script analyzes narrative-based cooperativity in myths from Donor Game simulations.
Adapted to handle N agents (unlike Trust Game which has exactly 2).

Key features:
- Supports arbitrary number of agents
- Pairwise lag correlation analysis between all agent pairs
- Population-level cooperativity metrics

The problems of current analysis/tokenization:
- Dictionary-based: Uses exact word matches (after tokenization/cleaning), not semantic analysis.
- No context: Doesn't consider negation (e.g., "not cooperative" would still count "cooperative").
- No stemming: Only exact matches (e.g., "cooperate" vs "cooperating").
"""

import json
import os
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np


# Common English stopwords
STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
    'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been', 'be', 'have', 'has', 'had',
    'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can',
    'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
    'them', 'their', 'his', 'her', 'its', 'our', 'your', 'who', 'which', 'what',
    'where', 'when', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
    'so', 'than', 'too', 'very', 's', 't', 'just', 'now', 'then'
}


def tokenize_and_clean(text: str) -> List[str]:
    """
    Tokenize text into words, remove stopwords and non-alphanumeric tokens.

    Args:
        text: Input text to tokenize

    Returns:
        List of cleaned tokens
    """
    # Convert to lowercase
    text = text.lower()

    # Extract words (alphanumeric sequences)
    words = re.findall(r'\b[a-z]+\b', text)

    # Remove stopwords and short words
    cleaned_words = [w for w in words if w not in STOPWORDS and len(w) > 2]

    return cleaned_words


def read_myths_from_json(filepath: str) -> Dict[str, Dict[int, str]]:
    """
    Read myths from Donor Game simulation state JSON file.

    Args:
        filepath: Path to the JSON file

    Returns:
        Dictionary mapping agent_id -> round -> myth_text
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    myths_by_agent = defaultdict(dict)

    # Extract myths from conversation history
    for entry in data.get('conversation_history', []):
        round_num = entry.get('round')
        myths = entry.get('myths', {})

        for agent_id, myth_text in myths.items():
            myths_by_agent[agent_id][round_num] = myth_text

    return dict(myths_by_agent)


def analyze_cooperativity(myths_by_agent: Dict[str, Dict[int, str]]) -> Dict[str, Dict[int, Dict[str, float]]]:
    """
    Analyze narrative-based cooperativity in myths.

    Args:
        myths_by_agent: Dictionary mapping agent_id -> round -> myth_text

    Returns:
        Dictionary mapping agent_id -> round -> cooperativity_scores
    """
    # Define cooperativity word categories
    collective_words = {'together', 'shared', 'mutual', 'partnership', 'community',
                       'collaboration', 'unity', 'collective', 'bond', 'alliance',
                       'cooperation', 'cooperate', 'collaborative'}

    individual_words = {'alone', 'solitary', 'independent', 'self', 'individual',
                       'isolated', 'separate', 'isolation', 'lonely', 'solo'}

    connected_words = {'relationship', 'connection', 'cooperation', 'reciprocity',
                      'trust', 'friendship', 'companion', 'ally', 'partner', 'kindness',
                      'generosity', 'compassion', 'solidarity'}

    disconnected_words = {'betrayal', 'isolation', 'division', 'conflict', 'rivalry',
                         'enemy', 'hostility', 'antagonism', 'opposition', 'discord'}

    giving_words = {'give', 'giving', 'gave', 'given', 'offered', 'offer', 'returned',
                   'return', 'shared', 'share', 'generous', 'gift', 'bestow', 'granted',
                   'donate', 'donated', 'donation', 'contribute', 'contributed'}

    taking_words = {'took', 'taken', 'take', 'kept', 'keep', 'withheld', 'withhold',
                   'refused', 'refuse', 'denied', 'deny', 'hoarded', 'hoard'}

    cooperativity_scores = {}

    for agent_id, rounds in myths_by_agent.items():
        cooperativity_scores[agent_id] = {}

        for round_num, myth_text in rounds.items():
            words = tokenize_and_clean(myth_text)

            # Count words in each category
            collective_count = sum(1 for w in words if w in collective_words)
            individual_count = sum(1 for w in words if w in individual_words)
            connected_count = sum(1 for w in words if w in connected_words)
            disconnected_count = sum(1 for w in words if w in disconnected_words)
            giving_count = sum(1 for w in words if w in giving_words)
            taking_count = sum(1 for w in words if w in taking_words)

            # Calculate cooperativity metrics
            cooperative_total = collective_count + connected_count + giving_count
            uncooperative_total = individual_count + disconnected_count + taking_count
            total_words = len(words)

            # Mutuality ratio (bounded, interpretable)
            mutuality_ratio = cooperative_total / (uncooperative_total + 1)

            # Cooperative proportion (as percentage of total words)
            cooperative_pct = (cooperative_total / total_words * 100) if total_words > 0 else 0
            uncooperative_pct = (uncooperative_total / total_words * 100) if total_words > 0 else 0

            cooperativity_scores[agent_id][round_num] = {
                'collective': collective_count,
                'individual': individual_count,
                'connected': connected_count,
                'disconnected': disconnected_count,
                'giving': giving_count,
                'taking': taking_count,
                'mutuality_ratio': mutuality_ratio,
                'cooperative_pct': cooperative_pct,
                'uncooperative_pct': uncooperative_pct,
                'total_words': total_words
            }

    return cooperativity_scores


def calculate_lag_correlation(scores_a: List[float], scores_b: List[float], lag: int = 1) -> float:
    """
    Calculate correlation between agent A's scores and agent B's lagged scores.

    Args:
        scores_a: Agent A's scores over time
        scores_b: Agent B's scores over time
        lag: How many rounds to lag (default 1 = previous round)

    Returns:
        Pearson correlation coefficient
    """
    if len(scores_a) <= lag or len(scores_b) <= lag:
        return 0.0

    # Agent A's scores from round lag+1 onwards
    a_lagged = scores_a[lag:]
    # Agent B's scores from round 1 to -lag
    b_lagged = scores_b[:-lag]

    if len(a_lagged) != len(b_lagged) or len(a_lagged) < 2:
        return 0.0

    # Calculate Pearson correlation
    mean_a = np.mean(a_lagged)
    mean_b = np.mean(b_lagged)

    numerator = np.sum((np.array(a_lagged) - mean_a) * (np.array(b_lagged) - mean_b))
    denominator = np.sqrt(np.sum((np.array(a_lagged) - mean_a)**2) * np.sum((np.array(b_lagged) - mean_b)**2))

    if denominator == 0:
        return 0.0

    return numerator / denominator


def calculate_all_lag_correlations(cooperativity_scores: Dict[str, Dict[int, Dict[str, float]]],
                                   metric: str = 'mutuality_ratio',
                                   lag: int = 1) -> Dict[Tuple[str, str], float]:
    """
    Calculate lag correlations for all pairs of agents.

    Args:
        cooperativity_scores: Cooperativity analysis results
        metric: Which metric to use for correlation
        lag: Lag in rounds

    Returns:
        Dictionary mapping (agent_a, agent_b) -> correlation
        Where agent_a adopts agent_b's behavior
    """
    agent_ids = sorted(cooperativity_scores.keys())
    correlations = {}

    for agent_a, agent_b in combinations(agent_ids, 2):
        # Get time series for each agent
        rounds_a = sorted(cooperativity_scores[agent_a].keys())
        rounds_b = sorted(cooperativity_scores[agent_b].keys())

        scores_a = [cooperativity_scores[agent_a][r][metric] for r in rounds_a]
        scores_b = [cooperativity_scores[agent_b][r][metric] for r in rounds_b]

        # Correlation: agent_a adopts agent_b's behavior
        corr_a_adopts_b = calculate_lag_correlation(scores_a, scores_b, lag=lag)
        # Correlation: agent_b adopts agent_a's behavior
        corr_b_adopts_a = calculate_lag_correlation(scores_b, scores_a, lag=lag)

        correlations[(agent_a, agent_b)] = corr_a_adopts_b
        correlations[(agent_b, agent_a)] = corr_b_adopts_a

    return correlations


def visualize_cooperativity(cooperativity_scores: Dict[str, Dict[int, Dict[str, float]]],
                            output_file: str = 'cooperativity_analysis.png'):
    """
    Visualize cooperativity measures for N agents.

    Args:
        cooperativity_scores: Cooperativity analysis results
        output_file: Output file path
    """
    agent_ids = sorted(cooperativity_scores.keys())
    n_agents = len(agent_ids)

    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, n_agents))
    agent_colors = {agent: colors[i] for i, agent in enumerate(agent_ids)}

    # Extract data for plotting
    agent_data = {}
    for agent_id in agent_ids:
        rounds = sorted(cooperativity_scores[agent_id].keys())
        agent_data[agent_id] = {
            'rounds': rounds,
            'mutuality_ratio': [cooperativity_scores[agent_id][r]['mutuality_ratio'] for r in rounds],
            'cooperative_pct': [cooperativity_scores[agent_id][r]['cooperative_pct'] for r in rounds],
            'uncooperative_pct': [cooperativity_scores[agent_id][r]['uncooperative_pct'] for r in rounds],
            'collective': [cooperativity_scores[agent_id][r]['collective'] for r in rounds],
            'connected': [cooperativity_scores[agent_id][r]['connected'] for r in rounds],
            'giving': [cooperativity_scores[agent_id][r]['giving'] for r in rounds],
        }

    # Determine grid size based on number of agents
    if n_agents <= 4:
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    else:
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Plot 1: Mutuality Ratio over time
    ax1 = fig.add_subplot(gs[0, 0])
    for agent_id in agent_ids:
        ax1.plot(agent_data[agent_id]['rounds'], agent_data[agent_id]['mutuality_ratio'],
                marker='o', linewidth=2, markersize=6, label=agent_id,
                color=agent_colors[agent_id], alpha=0.8)
    ax1.set_xlabel('Round (Generation)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mutuality Ratio', fontsize=12, fontweight='bold')
    ax1.set_title('Cooperativity: Mutuality Ratio Over Time', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9, loc='best')
    ax1.grid(True, alpha=0.3, linestyle='--')

    # Plot 2: Cooperative Language Percentage
    ax2 = fig.add_subplot(gs[0, 1])
    for agent_id in agent_ids:
        ax2.plot(agent_data[agent_id]['rounds'], agent_data[agent_id]['cooperative_pct'],
                marker='o', linewidth=2, markersize=6, label=agent_id,
                color=agent_colors[agent_id], alpha=0.8)
    ax2.set_xlabel('Round (Generation)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Percentage of Total Words', fontsize=12, fontweight='bold')
    ax2.set_title('Cooperative Language Percentage', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9, loc='best')
    ax2.grid(True, alpha=0.3, linestyle='--')

    # Plot 3: Population average cooperativity over time
    ax3 = fig.add_subplot(gs[1, 0])
    rounds = agent_data[agent_ids[0]]['rounds']
    avg_mutuality = []
    std_mutuality = []
    for i, r in enumerate(rounds):
        values = [agent_data[a]['mutuality_ratio'][i] for a in agent_ids if i < len(agent_data[a]['mutuality_ratio'])]
        avg_mutuality.append(np.mean(values))
        std_mutuality.append(np.std(values))

    ax3.plot(rounds, avg_mutuality, marker='s', linewidth=2.5, markersize=8,
            color='#2c3e50', label='Population Mean')
    ax3.fill_between(rounds,
                     np.array(avg_mutuality) - np.array(std_mutuality),
                     np.array(avg_mutuality) + np.array(std_mutuality),
                     alpha=0.3, color='#2c3e50')
    ax3.set_xlabel('Round (Generation)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Mutuality Ratio', fontsize=12, fontweight='bold')
    ax3.set_title('Population Average Cooperativity (±1 SD)', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')

    # Plot 4: Lag correlation heatmap
    ax4 = fig.add_subplot(gs[1, 1])
    correlations = calculate_all_lag_correlations(cooperativity_scores)

    # Build correlation matrix
    corr_matrix = np.zeros((n_agents, n_agents))
    for i, agent_a in enumerate(agent_ids):
        for j, agent_b in enumerate(agent_ids):
            if i != j:
                corr_matrix[i, j] = correlations.get((agent_a, agent_b), 0)

    im = ax4.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    ax4.set_xticks(range(n_agents))
    ax4.set_yticks(range(n_agents))
    ax4.set_xticklabels(agent_ids, rotation=45, ha='right')
    ax4.set_yticklabels(agent_ids)
    ax4.set_xlabel('Influencer (Round N-1)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Adopter (Round N)', fontsize=11, fontweight='bold')
    ax4.set_title('Lag-1 Mimicry Correlations', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax4, label='Correlation')

    # Add correlation values as text
    for i in range(n_agents):
        for j in range(n_agents):
            if i != j:
                text = ax4.text(j, i, f'{corr_matrix[i, j]:.2f}',
                               ha='center', va='center', fontsize=8,
                               color='white' if abs(corr_matrix[i, j]) > 0.5 else 'black')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Cooperativity visualization saved to {output_file}")
    plt.close()


def print_cooperativity_summary(cooperativity_scores: Dict[str, Dict[int, Dict[str, float]]]):
    """
    Print summary statistics for cooperativity analysis.

    Args:
        cooperativity_scores: Cooperativity analysis results
    """
    print("\n" + "="*80)
    print("DONOR GAME COOPERATIVITY ANALYSIS")
    print("="*80)

    agent_ids = sorted(cooperativity_scores.keys())

    for agent_id in agent_ids:
        print(f"\n{'─'*80}")
        print(f"AGENT: {agent_id}")
        print(f"{'─'*80}")

        rounds = sorted(cooperativity_scores[agent_id].keys())

        # Baseline (Round 1)
        baseline = cooperativity_scores[agent_id][rounds[0]]
        print(f"\nBaseline (Round {rounds[0]}):")
        print(f"  Mutuality Ratio: {baseline['mutuality_ratio']:.3f}")
        print(f"  Cooperative Language: {baseline['cooperative_pct']:.2f}%")
        print(f"  Uncooperative Language: {baseline['uncooperative_pct']:.2f}%")

        # Average across all rounds
        avg_mutuality = np.mean([cooperativity_scores[agent_id][r]['mutuality_ratio'] for r in rounds])
        avg_coop = np.mean([cooperativity_scores[agent_id][r]['cooperative_pct'] for r in rounds])
        avg_uncoop = np.mean([cooperativity_scores[agent_id][r]['uncooperative_pct'] for r in rounds])

        print(f"\nAverage across all rounds:")
        print(f"  Mutuality Ratio: {avg_mutuality:.3f}")
        print(f"  Cooperative Language: {avg_coop:.2f}%")
        print(f"  Uncooperative Language: {avg_uncoop:.2f}%")

    # Population-level statistics
    print(f"\n{'─'*80}")
    print("POPULATION-LEVEL STATISTICS")
    print(f"{'─'*80}")

    all_mutuality = []
    for agent_id in agent_ids:
        for r in cooperativity_scores[agent_id]:
            all_mutuality.append(cooperativity_scores[agent_id][r]['mutuality_ratio'])

    print(f"\nOverall Mutuality Ratio:")
    print(f"  Mean: {np.mean(all_mutuality):.3f}")
    print(f"  Std: {np.std(all_mutuality):.3f}")
    print(f"  Min: {np.min(all_mutuality):.3f}")
    print(f"  Max: {np.max(all_mutuality):.3f}")

    # Lag correlations
    print(f"\n{'─'*80}")
    print("LAG CORRELATIONS (Mimicry Analysis)")
    print(f"{'─'*80}")

    correlations = calculate_all_lag_correlations(cooperativity_scores)

    print("\nPairwise lag-1 correlations (row adopts column's behavior):")
    for (agent_a, agent_b), corr in sorted(correlations.items()):
        significance = ""
        if abs(corr) > 0.5:
            significance = " ***"
        elif abs(corr) > 0.3:
            significance = " **"
        elif abs(corr) > 0.1:
            significance = " *"
        print(f"  {agent_a} <- {agent_b}: r = {corr:.3f}{significance}")


def save_results_to_json(cooperativity_scores: Dict[str, Dict[int, Dict[str, float]]],
                         output_file: str):
    """
    Save cooperativity analysis results to JSON file.

    Args:
        cooperativity_scores: Cooperativity analysis results
        output_file: Output file path
    """
    # Convert int keys to strings for JSON serialization
    json_safe = {}
    for agent_id, rounds in cooperativity_scores.items():
        json_safe[agent_id] = {str(r): scores for r, scores in rounds.items()}

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_safe, f, indent=2, ensure_ascii=False)

    print(f"Cooperativity results saved to {output_file}")


def main():
    """Main execution function."""
    # Read myths from JSON
    print(f"Reading myths from {input_file}...")
    myths_by_agent = read_myths_from_json(input_file)

    print(f"Found {len(myths_by_agent)} agents")
    for agent_id, rounds in myths_by_agent.items():
        print(f"  {agent_id}: {len(rounds)} rounds")

    if not myths_by_agent:
        print("No myths found in the data. Exiting.")
        return

    # Analyze cooperativity
    print("\nAnalyzing cooperativity...")
    cooperativity_scores = analyze_cooperativity(myths_by_agent)

    # Print summary
    print_cooperativity_summary(cooperativity_scores)

    # Save results
    save_results_to_json(cooperativity_scores, output_file=json_output_file)

    # Visualize
    print("\nGenerating visualization...")
    visualize_cooperativity(cooperativity_scores, output_file=plot_output_file)

    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == '__main__':
    # Check if running from shell script (environment variables available)
    project_root = Path(__file__).resolve().parents[2]
    default_input = project_root / "data/json/donor_game/simulation_state.json"
    default_output = project_root / "data/plots/_local/donor_cooperativity_analysis"
    input_file = os.environ.get('ANALYSIS_INPUT_FILE', str(default_input))
    base_output_dir = os.environ.get('ANALYSIS_OUTPUT_DIR', str(default_output))
    task_name = os.environ.get('ANALYSIS_TASK_NAME', 'donor_game')
    os.makedirs(base_output_dir, exist_ok=True)

    # Output files
    json_output_file = f'{base_output_dir}/cooperativity_results.json'
    plot_output_file = f'{base_output_dir}/cooperativity_analysis.png'

    main()
