"""
Cross-condition analysis for djx grid experiments.

Scans all jobs from a djx run directory, extracts generosity metrics,
and produces comparison plots across grid dimensions.

Usage:
    python analyses_clean/DonorGame/analyze_djx_grid.py <djx_run_dir> [output_dir]

Example:
    python analyses_clean/DonorGame/analyze_djx_grid.py experiments/donor_game_myths/2026-01-30--12-16-08
"""

import sys
import os
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analyses_clean.DonorGame.trajectory_plotting import (
    load_simulation_data,
    calculate_trajectory_metrics,
    extract_per_agent_data,
)


def load_djx_results(run_dir: str) -> list:
    """
    Load all job results from a djx run directory.

    Returns list of dicts with keys: job_idx, labels, config, metrics, agent_data
    """
    run_path = Path(run_dir)
    results = []

    job_dirs = sorted(
        [d for d in run_path.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name),
    )

    for job_dir in job_dirs:
        config_file = job_dir / "config.yml"
        result_file = job_dir / "result.json"

        if not config_file.exists():
            print(f"  Skipping {job_dir.name}: no config.yml")
            continue

        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        labels = config.get('labels', {})
        task_order = config.get('simulation_params', {}).get('task_order', [])

        entry = {
            'job_idx': int(job_dir.name),
            'labels': labels,
            'config': config,
            'task_order': task_order,
            'has_game': 'game' in task_order,
            'has_myth': 'myth' in task_order,
            'metrics': None,
            'per_round_generosity': None,
        }

        if result_file.exists():
            try:
                data = load_simulation_data(str(result_file))
                conversation_history = data.get('conversation_history', [])
                game_data = data.get('game_data', {})

                if entry['has_game'] and conversation_history:
                    entry['metrics'] = calculate_trajectory_metrics(
                        conversation_history, game_data
                    )
                    # Extract per-round average generosity
                    agent_data = extract_per_agent_data(
                        conversation_history, game_data
                    )
                    if agent_data:
                        agents = sorted(agent_data.keys())
                        n_rounds = len(agent_data[agents[0]]['percentages'])
                        per_round_avg = []
                        for r in range(n_rounds):
                            round_pcts = [
                                agent_data[a]['percentages'][r]
                                for a in agents
                                if r < len(agent_data[a]['percentages'])
                            ]
                            per_round_avg.append(np.mean(round_pcts) if round_pcts else 0)
                        entry['per_round_generosity'] = per_round_avg
            except Exception as e:
                print(f"  Error loading {result_file}: {e}")
        else:
            print(f"  No result.json in job {job_dir.name} (not yet run?)")

        results.append(entry)

    return results


def _label_key(labels: dict) -> str:
    """Create a readable label from the labels dict."""
    parts = []
    for k, v in sorted(labels.items()):
        parts.append(f"{v}")
    return " | ".join(parts)


def plot_generosity_comparison(results: list, output_dir: str):
    """
    Bar chart of average generosity score per condition.
    Only includes jobs that have game data.
    """
    game_results = [r for r in results if r['metrics'] is not None]
    if not game_results:
        print("No game results to plot generosity comparison.")
        return

    game_results.sort(key=lambda r: _label_key(r['labels']))

    labels = [_label_key(r['labels']) for r in game_results]
    scores = [r['metrics']['generosity_score'] for r in game_results]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 6))
    bars = ax.bar(range(len(labels)), scores, color=plt.cm.Set2(
        np.linspace(0, 1, len(labels))
    ), edgecolor='gray', linewidth=0.5)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Average Generosity (%)', fontsize=12)
    ax.set_title('Average Generosity by Condition', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    save_path = f"{output_dir}/generosity_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_generosity_over_time(results: list, output_dir: str):
    """
    Line plot of population-average generosity (%) per round, one line per condition.
    Only includes jobs that have game data.
    """
    game_results = [r for r in results if r['per_round_generosity'] is not None]
    if not game_results:
        print("No game results to plot generosity over time.")
        return

    game_results.sort(key=lambda r: _label_key(r['labels']))

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(game_results)))

    for i, r in enumerate(game_results):
        rounds = list(range(1, len(r['per_round_generosity']) + 1))
        ax.plot(rounds, r['per_round_generosity'],
                marker='o', linewidth=2, markersize=4,
                label=_label_key(r['labels']),
                color=colors[i], alpha=0.85)

    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Population Average Generosity (%)', fontsize=12)
    ax.set_title('Generosity Over Time by Condition', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.legend(loc='best', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = f"{output_dir}/generosity_over_time.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_generosity_grouped(results: list, output_dir: str,
                            group_key: str, series_key: str):
    """
    Grouped bar chart: x-axis = group_key values, series = series_key values.
    Useful for e.g. group_key='myth_theme', series_key='task_order'.
    Only includes jobs that have game data.
    """
    game_results = [r for r in results if r['metrics'] is not None]
    if not game_results:
        print("No game results for grouped chart.")
        return

    # Check if the keys exist in labels
    if not all(group_key in r['labels'] and series_key in r['labels'] for r in game_results):
        print(f"Not all results have labels '{group_key}' and '{series_key}'. Skipping grouped chart.")
        return

    # Organize data
    groups = sorted(set(r['labels'][group_key] for r in game_results))
    series = sorted(set(r['labels'][series_key] for r in game_results))

    data = {}
    for r in game_results:
        key = (r['labels'][group_key], r['labels'][series_key])
        data[key] = r['metrics']['generosity_score']

    x = np.arange(len(groups))
    width = 0.8 / max(len(series), 1)
    colors = plt.cm.Set2(np.linspace(0, 1, len(series)))

    fig, ax = plt.subplots(figsize=(max(8, len(groups) * 1.5), 6))

    for i, s in enumerate(series):
        vals = [data.get((g, s), 0) for g in groups]
        offset = (i - len(series) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=s, color=colors[i],
                      edgecolor='gray', linewidth=0.5)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Average Generosity (%)', fontsize=12)
    ax.set_title(f'Generosity: {group_key} × {series_key}', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.legend(title=series_key, fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_path = f"{output_dir}/generosity_grouped_{group_key}_x_{series_key}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def save_summary_table(results: list, output_dir: str):
    """Save a JSON summary table of all conditions and their metrics."""
    rows = []
    for r in results:
        row = {
            'job_idx': r['job_idx'],
            **r['labels'],
            'task_order': r['task_order'],
            'has_game': r['has_game'],
        }
        if r['metrics']:
            row.update({
                'generosity_score': round(r['metrics']['generosity_score'], 2),
                'avg_donation_pct': round(r['metrics']['avg_donation_percentage'], 2),
                'total_donations': round(r['metrics']['total_donations'], 2),
                'final_balance_gini': round(r['metrics']['final_balance_gini'], 4),
                'total_rounds': r['metrics']['total_rounds'],
            })
        rows.append(row)

    save_path = f"{output_dir}/summary_table.json"
    with open(save_path, 'w') as f:
        json.dump(rows, f, indent=2)
    print(f"Saved: {save_path}")

    # Also print to console
    print("\n" + "=" * 90)
    print("SUMMARY TABLE")
    print("=" * 90)
    for row in rows:
        label = " | ".join(f"{v}" for k, v in sorted(row.items()) if k not in (
            'job_idx', 'task_order', 'has_game', 'total_donations',
            'total_rounds', 'final_balance_gini', 'avg_donation_pct',
        ))
        generosity = row.get('generosity_score', 'N/A (no game)')
        gini = row.get('final_balance_gini', '')
        gini_str = f"  Gini: {gini:.4f}" if isinstance(gini, float) else ""
        print(f"  Job {row['job_idx']:2d}  {label:50s}  Generosity: {generosity}{gini_str}")
    print("=" * 90)


def analyze_djx_grid(run_dir: str, output_dir: str = None):
    """Main entry point for djx grid analysis."""
    if output_dir is None:
        output_dir = str(Path(run_dir) / "_analysis")

    os.makedirs(output_dir, exist_ok=True)

    print(f"Analyzing djx grid run: {run_dir}")
    print(f"Output: {output_dir}")

    results = load_djx_results(run_dir)
    print(f"\nLoaded {len(results)} jobs")

    game_count = sum(1 for r in results if r['metrics'] is not None)
    myth_only_count = sum(1 for r in results if r['has_myth'] and not r['has_game'])
    print(f"  With game data: {game_count}")
    print(f"  Myth-only (no game): {myth_only_count}")

    # Summary table
    save_summary_table(results, output_dir)

    # Plots
    plot_generosity_comparison(results, output_dir)
    plot_generosity_over_time(results, output_dir)

    # Try grouped chart if we can detect two label dimensions
    all_label_keys = set()
    for r in results:
        all_label_keys.update(r['labels'].keys())

    label_keys = sorted(all_label_keys)
    if len(label_keys) >= 2:
        # Use the first two label keys for grouping
        plot_generosity_grouped(results, output_dir,
                                group_key=label_keys[0],
                                series_key=label_keys[1])

    print(f"\nAnalysis complete. Results in {output_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyses_clean/DonorGame/analyze_djx_grid.py <djx_run_dir> [output_dir]")
        print("\nExample:")
        print("  python analyses_clean/DonorGame/analyze_djx_grid.py experiments/donor_game_myths/2026-01-30--12-16-08")
        sys.exit(1)

    run_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    analyze_djx_grid(run_dir, output_dir)
