import json
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict, List, Tuple
from pathlib import Path

def load_simulation_data(filepath: str) -> Dict:
    """Load simulation state from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def calculate_trajectory_metrics(conversation_history: List[Dict]) -> Dict:
    """Calculate metrics to identify interesting trajectories"""
    metrics = {
        'sent_variance': np.var([r.get('sent', 0) for r in conversation_history]),
        'returned_variance': np.var([r.get('returned', 0) for r in conversation_history]),
        'investor_payoff_variance': np.var([r.get('investor_payoff', 0) for r in conversation_history]),
        'trustee_payoff_variance': np.var([r.get('trustee_payoff', 0) for r in conversation_history]),
        'total_variance': 0,
        'cooperation_score': 0,  # Average return ratio
        'trust_score': 0,  # Average send ratio
        'final_balance_diff': 0,
    }
    
    if conversation_history:
        sent_values = [r.get('sent', 0) for r in conversation_history]
        returned_values = [r.get('returned', 0) for r in conversation_history]
        received_values = [r.get('received', 0) for r in conversation_history]
        
        # Calculate cooperation (return ratio)
        return_ratios = [ret/rec if rec > 0 else 0 
                        for ret, rec in zip(returned_values, received_values)]
        metrics['cooperation_score'] = np.mean(return_ratios)
        
        # Calculate trust (send ratio of endowment)
        endowment = conversation_history[0].get('sent', 0) + conversation_history[0].get('investor_payoff', 0)
        if endowment > 0:
            send_ratios = [s/endowment for s in sent_values]
            metrics['trust_score'] = np.mean(send_ratios)
        
        # Final balance difference
        if 'balances' in conversation_history[-1]:
            balances = conversation_history[-1]['balances']
            if 'Agent_1' in balances and 'Agent_2' in balances:
                metrics['final_balance_diff'] = abs(balances['Agent_1'] - balances['Agent_2'])
        
        # Total variance (sum of all variances)
        metrics['total_variance'] = (
            metrics['sent_variance'] + 
            metrics['returned_variance'] + 
            metrics['investor_payoff_variance'] + 
            metrics['trustee_payoff_variance']
        )
    
    return metrics

def identify_interesting_trajectories(filepath: str, 
                                     top_n: int = 3,
                                     criteria: str = 'variance') -> List[Tuple[Dict, Dict]]:
    """
    Identify interesting trajectories based on various criteria
    
    Args:
        filepath: Path to simulation JSON file
        top_n: Number of interesting trajectories to return
        criteria: 'variance', 'cooperation', 'trust', 'balance_diff', or 'all'
    
    Returns:
        List of (trajectory_data, metrics) tuples
    """
    data = load_simulation_data(filepath)
    conversation_history = data.get('conversation_history', [])
    
    if not conversation_history:
        return []
    
    # Calculate metrics for this trajectory
    metrics = calculate_trajectory_metrics(conversation_history)
    
    # For now, return single trajectory (you can extend to compare multiple files)
    return [(conversation_history, metrics)]

def plot_numerical_trajectories(conversation_history: List[Dict], 
                                save_path: str = None,
                                title: str = "Trust Game Trajectory"):
    """
    Create comprehensive plots of numerical choices over rounds
    
    Args:
        conversation_history: List of round data dictionaries
        save_path: Optional path to save the figure
        title: Title for the plot
    """
    if not conversation_history:
        print("No conversation history to plot")
        return
    
    rounds = [r['round'] for r in conversation_history]
    
    # Extract role information to determine which agent had which role each round
    investor_agent_per_round = []
    trustee_agent_per_round = []
    for r in conversation_history:
        roles = r.get('roles', {})
        # Find which agent is investor and which is trustee
        investor_agent = None
        trustee_agent = None
        for agent_id, role in roles.items():
            if role == 'investor':
                investor_agent = agent_id
            elif role == 'trustee':
                trustee_agent = agent_id
        investor_agent_per_round.append(investor_agent)
        trustee_agent_per_round.append(trustee_agent)
    
    # Extract numerical data
    sent = [r.get('sent', 0) for r in conversation_history]
    received = [r.get('received', 0) for r in conversation_history]
    returned = [r.get('returned', 0) for r in conversation_history]
    investor_payoff = [r.get('investor_payoff', 0) for r in conversation_history]
    trustee_payoff = [r.get('trustee_payoff', 0) for r in conversation_history]
    
    # Extract balances
    investor_balance = []
    trustee_balance = []
    for r in conversation_history:
        balances = r.get('balances', {})
        investor_balance.append(balances.get('Agent_1', 0))
        trustee_balance.append(balances.get('Agent_2', 0))
    
    # Helper function to plot line with agent-specific markers
    def plot_with_agent_markers(ax, rounds, values, label, color, agent_per_round, linewidth=2):
        """Plot a line and overlay markers based on which agent had the role"""
        # Plot the line first
        if color:
            line = ax.plot(rounds, values, label=label, linewidth=linewidth, color=color)
            line_color = line[0].get_color()
        else:
            line = ax.plot(rounds, values, label=label, linewidth=linewidth)
            line_color = line[0].get_color()
        # Overlay markers: 'o' for Agent_1, '^' for Agent_2
        for i, agent in enumerate(agent_per_round):
            if agent:  # Only plot marker if agent is not None
                marker = 'o' if agent == 'Agent_1' else '^'
                ax.scatter(rounds[i], values[i], marker=marker, color=line_color, s=50, zorder=5)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Plot 1: Transaction amounts (sent, received, returned)
    ax1 = axes[0, 0]
    # Sent: marker based on which agent is investor (who sends)
    plot_with_agent_markers(ax1, rounds, sent, 'Sent', None, investor_agent_per_round)
    # Received: marker based on which agent is trustee (who receives)
    plot_with_agent_markers(ax1, rounds, received, 'Received', None, trustee_agent_per_round)
    # Returned: marker based on which agent is trustee (who returns)
    plot_with_agent_markers(ax1, rounds, returned, 'Returned', None, trustee_agent_per_round)
    ax1.set_xlabel('Round', fontsize=12)
    ax1.set_ylabel('Amount', fontsize=12)
    ax1.set_title('Transaction Flow', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(rounds)
    
    # Plot 2: Payoffs per round
    ax2 = axes[0, 1]
    # Investor Payoff: marker based on which agent is investor
    plot_with_agent_markers(ax2, rounds, investor_payoff, 'Investor Payoff', 
                            'green', investor_agent_per_round)
    # Trustee Payoff: marker based on which agent is trustee
    plot_with_agent_markers(ax2, rounds, trustee_payoff, 'Trustee Payoff', 
                            'orange', trustee_agent_per_round)
    ax2.set_xlabel('Round', fontsize=12)
    ax2.set_ylabel('Payoff', fontsize=12)
    ax2.set_title('Payoffs per Round', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(rounds)
    
    # Plot 3: Cumulative balances
    ax3 = axes[1, 0]
    # Investor Balance: marker based on which agent is investor in each round
    plot_with_agent_markers(ax3, rounds, investor_balance, 'Investor Balance', 
                            'blue', investor_agent_per_round)
    # Trustee Balance: marker based on which agent is trustee in each round
    plot_with_agent_markers(ax3, rounds, trustee_balance, 'Trustee Balance', 
                            'red', trustee_agent_per_round)
    ax3.set_xlabel('Round', fontsize=12)
    ax3.set_ylabel('Cumulative Balance', fontsize=12)
    ax3.set_title('Cumulative Balances', fontsize=14, fontweight='bold')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(rounds)
    
    # Plot 4: Return ratio and trust ratio
    ax4 = axes[1, 1]
    return_ratios = [ret/rec if rec > 0 else 0 
                    for ret, rec in zip(returned, received)]
    # Trust ratio (assuming endowment is constant, estimate from first round)
    endowment = sent[0] + investor_payoff[0] if len(sent) > 0 else 10
    trust_ratios = [s/endowment if endowment > 0 else 0 for s in sent]
    
    # Return Ratio: marker based on which agent is trustee (who returns)
    plot_with_agent_markers(ax4, rounds, return_ratios, 'Return Ratio', 
                            'purple', trustee_agent_per_round)
    # Trust Ratio: marker based on which agent is investor (who sends)
    plot_with_agent_markers(ax4, rounds, trust_ratios, 'Trust Ratio (Send/Endowment)', 
                            'brown', investor_agent_per_round)
    ax4.set_xlabel('Round', fontsize=12)
    ax4.set_ylabel('Ratio', fontsize=12)
    ax4.set_title('Cooperation & Trust Ratios', fontsize=14, fontweight='bold')
    ax4.legend(loc='best')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1.1)
    ax4.set_xticks(rounds)
    ax4.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Full Return/Trust')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved trajectory plot to {save_path}")
    else:
        plt.show()
    
    plt.close()

def display_raw_text_trajectory(conversation_history: List[Dict], 
                                agent_id: str = None,
                                max_rounds: int = 5):
    """
    Display raw text (myths) for a trajectory
    
    Args:
        conversation_history: List of round data dictionaries
        agent_id: Specific agent to show (None for both)
        max_rounds: Maximum number of rounds to display
    """
    print("\n" + "=" * 80)
    print("RAW TEXT TRAJECTORY (MYTHS)")
    print("=" * 80)
    
    rounds_to_show = conversation_history[:max_rounds]
    
    for round_data in rounds_to_show:
        round_num = round_data.get('round', '?')
        myths = round_data.get('myths', {})
        
        print(f"\n{'='*80}")
        print(f"ROUND {round_num}")
        print(f"{'='*80}")
        
        # Display numerical summary
        print(f"\nNumerical Summary:")
        print(f"  Sent: {round_data.get('sent', 0)}")
        print(f"  Received: {round_data.get('received', 0)}")
        print(f"  Returned: {round_data.get('returned', 0)}")
        print(f"  Investor Payoff: {round_data.get('investor_payoff', 0)}")
        print(f"  Trustee Payoff: {round_data.get('trustee_payoff', 0)}")
        
        # Display myths
        print(f"\n{'─'*80}")
        print("MYTHS:")
        print(f"{'─'*80}")
        
        if agent_id and agent_id in myths:
            print(f"\n{agent_id}:")
            print(f"{'─'*40}")
            print(myths[agent_id])
        else:
            for agent, myth_text in myths.items():
                print(f"\n{agent}:")
                print(f"{'─'*40}")
                print(myth_text)
                print()

def analyze_interesting_trajectories(filepath: str,
                                     output_dir: str,
                                     top_n: int = 3):
    """
    Complete analysis: identify, visualize, and display interesting trajectories

    Args:
        filepath: Path to simulation JSON file
        output_dir: Directory to save plots
        top_n: Number of trajectories to analyze
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Identify interesting trajectories
    trajectories = identify_interesting_trajectories(filepath, top_n=top_n)

    if not trajectories:
        print("No trajectories found")
        return

    for idx, (conversation_history, metrics) in enumerate(trajectories):
        print(f"\n{'='*80}")
        print(f"TRAJECTORY {idx + 1}")
        print(f"{'='*80}")

        # Print metrics
        print("\nTrajectory Metrics:")
        print(f"  Total Variance: {metrics['total_variance']:.2f}")
        print(f"  Cooperation Score (avg return ratio): {metrics['cooperation_score']:.3f}")
        print(f"  Trust Score (avg send ratio): {metrics['trust_score']:.3f}")
        print(f"  Final Balance Difference: {metrics['final_balance_diff']:.2f}")

        # Generate plot
        plot_filename = f"{output_dir}/trajectory_{idx+1}_numerical.png"
        plot_numerical_trajectories(
            conversation_history,
            save_path=plot_filename,
            title=f"Trajectory {idx+1} - Numerical Choices"
        )

        # Display raw text
        display_raw_text_trajectory(conversation_history, max_rounds=10)

    print(f"\n{'='*80}")
    print("Analysis complete!")
    print(f"file saved to {output_dir}")
    print(f"{'='*80}")

def main():
    """Main function to run trajectory analysis"""
    # Run analysis
    analyze_interesting_trajectories(
        filepath=input_filepath,
        output_dir=output_dir,
        top_n=number_of_trajectories
    )

if __name__ == "__main__":
    # Check if running from shell script (environment variables available)
    project_root = Path(__file__).resolve().parents[1]
    default_input = project_root / "data/json/main_loop/task_order/game_myth.json"
    default_output = project_root / "data/plots/_local/trajectory_plotting"
    input_filepath = os.environ.get('ANALYSIS_INPUT_FILE', str(default_input))
    output_dir = os.environ.get('ANALYSIS_OUTPUT_DIR', str(default_output))
    task_name = os.environ.get('ANALYSIS_TASK_NAME', 'game_myth')
    number_of_trajectories = 1
    os.makedirs(output_dir, exist_ok=True)

    main()