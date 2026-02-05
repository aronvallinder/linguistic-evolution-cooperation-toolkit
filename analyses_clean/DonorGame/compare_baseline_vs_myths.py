"""
Compare Baseline (game_only) vs Myth-Primed Conditions

Creates comparison plots showing how myth-priming affects donor game behavior.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_baseline_results(baseline_path: str) -> dict:
    """Load and calculate metrics from baseline result.json"""
    with open(baseline_path) as f:
        data = json.load(f)

    game = data.get('game_data', {})
    donations = game.get('donation_history', [])
    balances = game.get('balances', {})

    if not donations:
        return None

    percentages = [d['percentage'] for d in donations]

    # Gini coefficient
    vals = sorted(balances.values())
    n = len(vals)
    weighted_sum = sum((i + 1) * v for i, v in enumerate(vals))
    total = sum(vals)
    gini = (2 * weighted_sum) / (n * total) - (n + 1) / n
    gini = max(0, gini)

    return {
        'condition': 'baseline (game_only)',
        'myth_theme': 'none',
        'generosity_score': np.mean(percentages),
        'total_donations': sum(d['amount'] for d in donations),
        'final_balance_gini': gini,
        'total_rounds': len(set(d['round'] for d in donations))
    }


def load_myths_summary(summary_path: str) -> list:
    """Load myth experiment summary table"""
    with open(summary_path) as f:
        data = json.load(f)

    # Filter to only myth+game conditions
    return [d for d in data if d.get('has_game', False)]


def plot_comparison(baseline: dict, myth_conditions: list, output_path: str):
    """Create comparison bar chart"""

    # Prepare data
    conditions = ['baseline\n(no myth)'] + [c['myth_theme'].replace('_', '\n') for c in myth_conditions]
    generosity = [baseline['generosity_score']] + [c['generosity_score'] for c in myth_conditions]
    gini = [baseline['final_balance_gini']] + [c['final_balance_gini'] for c in myth_conditions]

    # Colors: baseline in gray, myths in color gradient
    colors = ['#6c757d'] + list(plt.cm.Set2(np.linspace(0, 1, len(myth_conditions))))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Donor Game: Baseline vs Myth-Primed Conditions', fontsize=14, fontweight='bold')

    # Plot 1: Generosity comparison
    ax1 = axes[0]
    bars1 = ax1.bar(conditions, generosity, color=colors, edgecolor='black', linewidth=0.5)
    ax1.axhline(y=baseline['generosity_score'], color='#6c757d', linestyle='--', alpha=0.7,
                label=f'Baseline: {baseline["generosity_score"]:.1f}%')
    ax1.set_ylabel('Average Generosity (%)', fontsize=12)
    ax1.set_title('Generosity by Condition', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 60)
    ax1.legend(loc='upper right')

    # Add value labels
    for bar, val in zip(bars1, generosity):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10)

    # Add delta labels for myth conditions
    for i, (bar, val) in enumerate(zip(bars1[1:], generosity[1:]), 1):
        delta = val - baseline['generosity_score']
        color = 'green' if delta > 0 else 'red'
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                f'({delta:+.1f})', ha='center', va='bottom', fontsize=9, color=color)

    # Plot 2: Gini (inequality) comparison
    ax2 = axes[1]
    bars2 = ax2.bar(conditions, gini, color=colors, edgecolor='black', linewidth=0.5)
    ax2.axhline(y=baseline['final_balance_gini'], color='#6c757d', linestyle='--', alpha=0.7,
                label=f'Baseline: {baseline["final_balance_gini"]:.3f}')
    ax2.set_ylabel('Final Balance Gini (lower = more equal)', fontsize=12)
    ax2.set_title('Wealth Inequality by Condition', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 0.035)
    ax2.legend(loc='upper right')

    # Add value labels
    for bar, val in zip(bars2, gini):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to {output_path}")
    plt.close()


def create_summary_table(baseline: dict, myth_conditions: list, output_path: str):
    """Create combined summary table JSON"""

    all_conditions = [baseline] + myth_conditions

    with open(output_path, 'w') as f:
        json.dump(all_conditions, f, indent=2)

    print(f"Saved combined summary to {output_path}")


def main(baseline_result_path: str, myths_summary_path: str, output_dir: str):
    """Run the comparison analysis"""

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load data
    baseline = load_baseline_results(baseline_result_path)
    myth_conditions = load_myths_summary(myths_summary_path)

    if not baseline:
        print("Error: Could not load baseline results")
        return

    print(f"Loaded baseline: {baseline['generosity_score']:.2f}% generosity")
    print(f"Loaded {len(myth_conditions)} myth conditions")

    # Generate outputs
    plot_comparison(baseline, myth_conditions, f"{output_dir}/baseline_vs_myths_comparison.png")
    create_summary_table(baseline, myth_conditions, f"{output_dir}/combined_summary.json")

    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY: Baseline vs Myth-Primed Conditions")
    print("="*70)
    print(f"{'Condition':<30} {'Generosity':>12} {'Δ Baseline':>12} {'Gini':>10}")
    print("-"*70)
    print(f"{'baseline (game_only)':<30} {baseline['generosity_score']:>11.2f}% {'-':>12} {baseline['final_balance_gini']:>10.4f}")

    for cond in myth_conditions:
        delta = cond['generosity_score'] - baseline['generosity_score']
        delta_str = f"{delta:+.2f}%"
        print(f"{cond['myth_theme']:<30} {cond['generosity_score']:>11.2f}% {delta_str:>12} {cond['final_balance_gini']:>10.4f}")

    print("="*70)


if __name__ == "__main__":
    import sys
    import os

    # Default paths
    project_root = Path(__file__).resolve().parents[2]

    baseline_path = os.environ.get(
        'BASELINE_RESULT_PATH',
        str(project_root / "experiments/donor_game_baseline/2026-02-05--10-26-24/0/result.json")
    )
    myths_summary_path = os.environ.get(
        'MYTHS_SUMMARY_PATH',
        str(project_root / "experiments/donor_game_myths/2026-02-02--10-45-36/_analysis/summary_table.json")
    )
    output_dir = os.environ.get(
        'OUTPUT_DIR',
        str(project_root / "experiments/donor_game_myths/2026-02-02--10-45-36/_analysis")
    )

    main(baseline_path, myths_summary_path, output_dir)
