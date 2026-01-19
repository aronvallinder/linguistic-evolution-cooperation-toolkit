"""
Trajectory Plotting for Donor Game Simulations

This script visualizes game round outcomes for the Donor Game, including:
- Donation amounts and percentages
- Balance trajectories for all agents
- Generosity metrics over time
- Pairwise interaction patterns
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict, List, Tuple
from pathlib import Path
from collections import defaultdict


def load_simulation_data(filepath: str) -> Dict:
    """Load simulation state from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def calculate_trajectory_metrics(conversation_history: List[Dict], game_data: Dict) -> Dict:
    """
    Calculate metrics to characterize Donor Game trajectories.

    Args:
        conversation_history: List of round data dictionaries
        game_data: Game-level data including balances and donation_history

    Returns:
        Dictionary of trajectory metrics
    """
    donation_history = game_data.get('donation_history', [])
    balances = game_data.get('balances', {})

    metrics = {
        'total_donations': 0,
        'avg_donation_percentage': 0,
        'donation_variance': 0,
        'balance_variance': 0,
        'generosity_score': 0,  # Average donation percentage
        'final_balance_gini': 0,  # Inequality measure
        'total_rounds': len(conversation_history),
    }

    if donation_history:
        amounts = [d['amount'] for d in donation_history]
        percentages = [d['percentage'] for d in donation_history]

        metrics['total_donations'] = sum(amounts)
        metrics['avg_donation_percentage'] = np.mean(percentages)
        metrics['donation_variance'] = np.var(percentages)
        metrics['generosity_score'] = np.mean(percentages)

    if balances:
        balance_values = list(balances.values())
        metrics['balance_variance'] = np.var(balance_values)

        # Calculate Gini coefficient for final balance inequality
        if len(balance_values) > 1:
            sorted_balances = sorted(balance_values)
            n = len(sorted_balances)
            cumulative = np.cumsum(sorted_balances)
            gini = (2 * np.sum((np.arange(1, n + 1) * sorted_balances))) / (n * np.sum(sorted_balances)) - (n + 1) / n
            metrics['final_balance_gini'] = max(0, gini)

    return metrics


def extract_per_agent_data(conversation_history: List[Dict], game_data: Dict) -> Dict[str, Dict]:
    """
    Extract per-agent donation and balance data across rounds.

    Returns:
        Dictionary mapping agent_id -> {rounds, donations, percentages, balances, received}
    """
    donation_history = game_data.get('donation_history', [])

    # Get all unique agents
    agents = set()
    for entry in conversation_history:
        if 'balances' in entry:
            agents.update(entry['balances'].keys())

    agent_data = {agent: {
        'rounds': [],
        'donations': [],
        'percentages': [],
        'balances': [],
        'received': [],
    } for agent in agents}

    # Track donations and receipts per round per agent
    donations_by_round = defaultdict(lambda: defaultdict(float))
    percentages_by_round = defaultdict(lambda: defaultdict(float))
    received_by_round = defaultdict(lambda: defaultdict(float))

    for donation in donation_history:
        round_num = donation['round']
        donor = donation['donor_id']
        recipient = donation['recipient_id']
        amount = donation['amount']
        percentage = donation['percentage']

        donations_by_round[round_num][donor] = amount
        percentages_by_round[round_num][donor] = percentage
        # Received amount includes multiplier effect (stored in history as the donated amount)
        # The actual received amount would be amount * multiplier, but we track donated amount here
        received_by_round[round_num][recipient] += amount

    # Extract balance trajectory from conversation history
    for entry in conversation_history:
        round_num = entry['round']
        balances = entry.get('balances', {})

        for agent in agents:
            agent_data[agent]['rounds'].append(round_num)
            agent_data[agent]['donations'].append(donations_by_round[round_num].get(agent, 0))
            agent_data[agent]['percentages'].append(percentages_by_round[round_num].get(agent, 0))
            agent_data[agent]['balances'].append(balances.get(agent, 0))
            agent_data[agent]['received'].append(received_by_round[round_num].get(agent, 0))

    return agent_data


def plot_numerical_trajectories(conversation_history: List[Dict],
                                game_data: Dict,
                                save_path: str = None,
                                title: str = "Donor Game Trajectory"):
    """
    Create comprehensive plots of Donor Game dynamics over rounds.

    Args:
        conversation_history: List of round data dictionaries
        game_data: Game-level data including balances and donation_history
        save_path: Optional path to save the figure
        title: Title for the plot
    """
    if not conversation_history:
        print("No conversation history to plot")
        return

    agent_data = extract_per_agent_data(conversation_history, game_data)
    agents = sorted(agent_data.keys())

    if not agents:
        print("No agent data found")
        return

    # Color palette for multiple agents
    colors = plt.cm.tab10(np.linspace(0, 1, len(agents)))
    agent_colors = {agent: colors[i] for i, agent in enumerate(agents)}

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Plot 1: Donation amounts over time
    ax1 = axes[0, 0]
    for agent in agents:
        rounds = agent_data[agent]['rounds']
        donations = agent_data[agent]['donations']
        ax1.plot(rounds, donations, marker='o', linewidth=2, markersize=6,
                label=agent, color=agent_colors[agent], alpha=0.8)
    ax1.set_xlabel('Round', fontsize=12)
    ax1.set_ylabel('Donation Amount', fontsize=12)
    ax1.set_title('Donation Amounts Over Time', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Donation percentages over time
    ax2 = axes[0, 1]
    for agent in agents:
        rounds = agent_data[agent]['rounds']
        percentages = agent_data[agent]['percentages']
        ax2.plot(rounds, percentages, marker='s', linewidth=2, markersize=6,
                label=agent, color=agent_colors[agent], alpha=0.8)
    ax2.set_xlabel('Round', fontsize=12)
    ax2.set_ylabel('Donation Percentage (%)', fontsize=12)
    ax2.set_title('Generosity (% of Resources Donated)', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 105)
    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% line')

    # Plot 3: Balance trajectories
    ax3 = axes[1, 0]
    for agent in agents:
        rounds = agent_data[agent]['rounds']
        balances = agent_data[agent]['balances']
        ax3.plot(rounds, balances, marker='^', linewidth=2, markersize=6,
                label=agent, color=agent_colors[agent], alpha=0.8)
    ax3.set_xlabel('Round', fontsize=12)
    ax3.set_ylabel('Balance', fontsize=12)
    ax3.set_title('Resource Balances Over Time', fontsize=14, fontweight='bold')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Net flow (received - donated) per round
    ax4 = axes[1, 1]
    for agent in agents:
        rounds = agent_data[agent]['rounds']
        donations = agent_data[agent]['donations']
        received = agent_data[agent]['received']
        # Note: received here is the raw donated amount, actual received = received * multiplier
        # For visualization, we show the donation flow pattern
        net_flow = [r - d for r, d in zip(received, donations)]
        ax4.plot(rounds, net_flow, marker='d', linewidth=2, markersize=6,
                label=agent, color=agent_colors[agent], alpha=0.8)
    ax4.set_xlabel('Round', fontsize=12)
    ax4.set_ylabel('Net Flow (Received - Donated)', fontsize=12)
    ax4.set_title('Net Resource Flow Per Round', fontsize=14, fontweight='bold')
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved trajectory plot to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_aggregate_statistics(conversation_history: List[Dict],
                              game_data: Dict,
                              save_path: str = None,
                              title: str = "Donor Game Aggregate Statistics"):
    """
    Create aggregate statistics plots for the Donor Game.

    Args:
        conversation_history: List of round data dictionaries
        game_data: Game-level data
        save_path: Optional path to save the figure
        title: Title for the plot
    """
    agent_data = extract_per_agent_data(conversation_history, game_data)
    agents = sorted(agent_data.keys())

    if not agents:
        print("No agent data found")
        return

    colors = plt.cm.tab10(np.linspace(0, 1, len(agents)))
    agent_colors = {agent: colors[i] for i, agent in enumerate(agents)}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Plot 1: Average donation percentage per agent (bar chart)
    ax1 = axes[0, 0]
    avg_percentages = [np.mean(agent_data[a]['percentages']) for a in agents]
    bars = ax1.bar(agents, avg_percentages, color=[agent_colors[a] for a in agents], alpha=0.8)
    ax1.set_ylabel('Average Donation %', fontsize=12)
    ax1.set_title('Average Generosity by Agent', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 100)
    for bar, val in zip(bars, avg_percentages):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
    ax1.tick_params(axis='x', rotation=45)

    # Plot 2: Final balances (bar chart)
    ax2 = axes[0, 1]
    final_balances = [agent_data[a]['balances'][-1] if agent_data[a]['balances'] else 0 for a in agents]
    bars = ax2.bar(agents, final_balances, color=[agent_colors[a] for a in agents], alpha=0.8)
    ax2.set_ylabel('Final Balance', fontsize=12)
    ax2.set_title('Final Resource Balances', fontsize=14, fontweight='bold')
    for bar, val in zip(bars, final_balances):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.0f}', ha='center', va='bottom', fontsize=10)
    ax2.tick_params(axis='x', rotation=45)

    # Plot 3: Donation percentage distribution (box plot)
    ax3 = axes[1, 0]
    donation_data = [agent_data[a]['percentages'] for a in agents]
    bp = ax3.boxplot(donation_data, labels=agents, patch_artist=True)
    for patch, agent in zip(bp['boxes'], agents):
        patch.set_facecolor(agent_colors[agent])
        patch.set_alpha(0.7)
    ax3.set_ylabel('Donation Percentage', fontsize=12)
    ax3.set_title('Donation Percentage Distribution', fontsize=14, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)

    # Plot 4: Cumulative donations over time (stacked area or line)
    ax4 = axes[1, 1]
    rounds = agent_data[agents[0]]['rounds'] if agents else []
    for agent in agents:
        cumulative = np.cumsum(agent_data[agent]['donations'])
        ax4.plot(rounds, cumulative, marker='o', linewidth=2, markersize=4,
                label=agent, color=agent_colors[agent], alpha=0.8)
    ax4.set_xlabel('Round', fontsize=12)
    ax4.set_ylabel('Cumulative Donations', fontsize=12)
    ax4.set_title('Cumulative Donations Over Time', fontsize=14, fontweight='bold')
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved aggregate statistics plot to {save_path}")
    else:
        plt.show()

    plt.close()


def display_raw_text_trajectory(conversation_history: List[Dict],
                                agent_id: str = None,
                                max_rounds: int = 5):
    """
    Display raw text (myths) for a trajectory.

    Args:
        conversation_history: List of round data dictionaries
        agent_id: Specific agent to show (None for all)
        max_rounds: Maximum number of rounds to display
    """
    print("\n" + "=" * 80)
    print("RAW TEXT TRAJECTORY (MYTHS)")
    print("=" * 80)

    rounds_to_show = conversation_history[:max_rounds]

    for round_data in rounds_to_show:
        round_num = round_data.get('round', '?')
        myths = round_data.get('myths', {})
        donations = round_data.get('donations', [])
        balances = round_data.get('balances', {})

        print(f"\n{'='*80}")
        print(f"ROUND {round_num}")
        print(f"{'='*80}")

        # Display donation summary
        print(f"\nDonation Summary:")
        if donations:
            for d in donations:
                print(f"  {d['donor_id']} -> {d['recipient_id']}: {d['amount']:.1f} ({d['percentage']:.1f}%)")
        else:
            print("  No donation data available")

        # Display balances
        print(f"\nBalances after round:")
        for agent, balance in sorted(balances.items()):
            print(f"  {agent}: {balance:.1f}")

        # Display myths
        if myths:
            print(f"\n{'─'*80}")
            print("MYTHS:")
            print(f"{'─'*80}")

            if agent_id and agent_id in myths:
                print(f"\n{agent_id}:")
                print(f"{'─'*40}")
                print(myths[agent_id])
            else:
                for agent, myth_text in sorted(myths.items()):
                    print(f"\n{agent}:")
                    print(f"{'─'*40}")
                    print(myth_text)
                    print()


def analyze_donor_trajectories(filepath: str,
                               output_dir: str,
                               top_n: int = 1):
    """
    Complete analysis: visualize and display Donor Game trajectories.

    Args:
        filepath: Path to simulation JSON file
        output_dir: Directory to save plots
        top_n: Number of trajectories to analyze (for future multi-file support)
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load data
    data = load_simulation_data(filepath)
    conversation_history = data.get('conversation_history', [])
    game_data = data.get('game_data', {})

    if not conversation_history:
        print("No conversation history found")
        return

    # Calculate metrics
    metrics = calculate_trajectory_metrics(conversation_history, game_data)

    print(f"\n{'='*80}")
    print("DONOR GAME TRAJECTORY ANALYSIS")
    print(f"{'='*80}")

    # Print metrics
    print("\nTrajectory Metrics:")
    print(f"  Total Rounds: {metrics['total_rounds']}")
    print(f"  Total Donations: {metrics['total_donations']:.2f}")
    print(f"  Average Donation %: {metrics['avg_donation_percentage']:.2f}%")
    print(f"  Generosity Score: {metrics['generosity_score']:.2f}%")
    print(f"  Donation Variance: {metrics['donation_variance']:.2f}")
    print(f"  Final Balance Gini: {metrics['final_balance_gini']:.3f}")

    # Generate plots
    plot_filename = f"{output_dir}/donor_trajectory_numerical.png"
    plot_numerical_trajectories(
        conversation_history,
        game_data,
        save_path=plot_filename,
        title="Donor Game - Numerical Trajectories"
    )

    aggregate_filename = f"{output_dir}/donor_trajectory_aggregate.png"
    plot_aggregate_statistics(
        conversation_history,
        game_data,
        save_path=aggregate_filename,
        title="Donor Game - Aggregate Statistics"
    )

    # Display raw text
    display_raw_text_trajectory(conversation_history, max_rounds=10)

    print(f"\n{'='*80}")
    print("Analysis complete!")
    print(f"Plots saved to {output_dir}")
    print(f"{'='*80}")


def main():
    """Main function to run trajectory analysis"""
    analyze_donor_trajectories(
        filepath=input_filepath,
        output_dir=output_dir,
        top_n=number_of_trajectories
    )


if __name__ == "__main__":
    # Check if running from shell script (environment variables available)
    project_root = Path(__file__).resolve().parents[2]
    default_input = project_root / "data/json/donor_game/simulation_state.json"
    default_output = project_root / "data/plots/_local/donor_trajectory_plotting"
    input_filepath = os.environ.get('ANALYSIS_INPUT_FILE', str(default_input))
    output_dir = os.environ.get('ANALYSIS_OUTPUT_DIR', str(default_output))
    task_name = os.environ.get('ANALYSIS_TASK_NAME', 'donor_game')
    number_of_trajectories = 1
    os.makedirs(output_dir, exist_ok=True)

    main()
