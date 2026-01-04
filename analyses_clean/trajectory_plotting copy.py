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
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Plot 1: Transaction amounts (sent, received, returned)
    ax1 = axes[0, 0]
    ax1.plot(rounds, sent, marker='o', label='Sent', linewidth=2, markersize=8)
    ax1.plot(rounds, received, marker='s', label='Received', linewidth=2, markersize=8)
    ax1.plot(rounds, returned, marker='^', label='Returned', linewidth=2, markersize=8)
    ax1.set_xlabel('Round', fontsize=12)
    ax1.set_ylabel('Amount', fontsize=12)
    ax1.set_title('Transaction Flow', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(rounds)
    
    # Plot 2: Payoffs per round
    ax2 = axes[0, 1]
    ax2.plot(rounds, investor_payoff, marker='o', label='Investor Payoff', 
             linewidth=2, markersize=8, color='green')
    ax2.plot(rounds, trustee_payoff, marker='s', label='Trustee Payoff', 
             linewidth=2, markersize=8, color='orange')
    ax2.set_xlabel('Round', fontsize=12)
    ax2.set_ylabel('Payoff', fontsize=12)
    ax2.set_title('Payoffs per Round', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(rounds)
    
    # Plot 3: Cumulative balances
    ax3 = axes[1, 0]
    ax3.plot(rounds, investor_balance, marker='o', label='Investor Balance', 
             linewidth=2, markersize=8, color='blue')
    ax3.plot(rounds, trustee_balance, marker='s', label='Trustee Balance', 
             linewidth=2, markersize=8, color='red')
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
    
    ax4.plot(rounds, return_ratios, marker='o', label='Return Ratio', 
             linewidth=2, markersize=8, color='purple')
    ax4.plot(rounds, trust_ratios, marker='s', label='Trust Ratio (Send/Endowment)', 
             linewidth=2, markersize=8, color='brown')
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