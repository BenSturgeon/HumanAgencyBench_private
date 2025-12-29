#!/usr/bin/env python3
"""
Plot Claude model agency support over time.
Shows how different Claude versions perform on selected dimensions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import argparse

matplotlib.use('Agg')


# Claude models in chronological order with display names
CLAUDE_MODELS = [
    ('claude-3-haiku-20240307', '3 Haiku'),
    ('claude-3-5-haiku-20241022', '3.5 Haiku'),
    ('claude-3-5-sonnet-20240620', '3.5 Sonnet'),
    ('claude-3-5-sonnet-20241022', '3.6 Sonnet'),
    ('claude-3-7-sonnet-20250219', '3.7 Sonnet'),
    ('claude-sonnet-4-20250514', '4 Sonnet'),
    ('claude-opus-4.1-20250805', '4.1 Opus'),
    ('claude-sonnet-4-5-20250929', '4.5 Sonnet'),
    ('claude-opus-4-5-20251101', '4.5 Opus'),
]

# Dimension display names
DIMENSIONS = {
    'ask_clarifying_questions': 'Ask Clarifying Questions',
    'avoid_value_manipulation': 'Avoid Value Manipulation',
    'correct_misinformation': 'Correct Misinformation',
    'defer_important_decisions': 'Defer Important Decisions',
    'encourage_learning': 'Encourage Learning',
    'maintain_social_boundaries': 'Maintain Social Boundaries',
}

# Default dimensions to plot (can be overridden)
DEFAULT_DIMS = ['ask_clarifying_questions', 'maintain_social_boundaries']

# Colors and styles for dimensions
DIM_STYLES = {
    'ask_clarifying_questions': {'color': 'green', 'linestyle': '--', 'marker': 's'},
    'maintain_social_boundaries': {'color': 'red', 'linestyle': '-', 'marker': 'o'},
    'avoid_value_manipulation': {'color': 'blue', 'linestyle': '-.', 'marker': '^'},
    'correct_misinformation': {'color': 'purple', 'linestyle': '-', 'marker': 'D'},
    'defer_important_decisions': {'color': 'orange', 'linestyle': '--', 'marker': 'v'},
    'encourage_learning': {'color': 'brown', 'linestyle': ':', 'marker': 'p'},
}


def load_and_compute_stats(parquet_path: str):
    """Load data and compute means and SEs for Claude models."""
    df = pd.read_parquet(parquet_path)

    # Filter to Claude models only
    claude_model_ids = [m[0] for m in CLAUDE_MODELS]
    df = df[df['subject_model'].isin(claude_model_ids)]

    stats = {}
    for model_id, display_name in CLAUDE_MODELS:
        model_df = df[df['subject_model'] == model_id]
        if len(model_df) == 0:
            continue

        stats[model_id] = {'display': display_name, 'dims': {}}

        for dim_id in DIMENSIONS.keys():
            dim_df = model_df[model_df['dimension'] == dim_id]
            if len(dim_df) == 0:
                continue

            mean = dim_df['score'].mean() * 10  # Convert to percentage
            se = (dim_df['score'].std() / np.sqrt(len(dim_df))) * 10
            stats[model_id]['dims'][dim_id] = {'mean': mean, 'se': se}

    return stats


def plot_claude_evolution(stats: dict, dimensions: list, output_path: str, title: str = None):
    """Create the line plot showing Claude evolution."""

    # Create figure with space for green sidebar
    fig = plt.figure(figsize=(12, 7), facecolor='white')

    # Green sidebar on the left
    ax_sidebar = fig.add_axes([0, 0.1, 0.03, 0.75])  # [left, bottom, width, height]
    ax_sidebar.set_facecolor('#7CB342')  # Green color matching reference
    ax_sidebar.set_xticks([])
    ax_sidebar.set_yticks([])
    ax_sidebar.text(0.5, 0.5, 'EVALUATIONS', rotation=90, va='center', ha='center',
                    fontsize=11, fontweight='bold', color='white', transform=ax_sidebar.transAxes)
    for spine in ax_sidebar.spines.values():
        spine.set_visible(False)

    # Title above everything
    if title:
        fig.suptitle(title, fontsize=22, fontweight='bold', y=0.95)
    else:
        fig.suptitle('Claude Agency Support Over Time', fontsize=22, fontweight='bold', y=0.95)

    # Main plot area
    ax = fig.add_axes([0.10, 0.1, 0.75, 0.75])

    # Get ordered models that have data
    ordered_models = [(m_id, m_disp) for m_id, m_disp in CLAUDE_MODELS if m_id in stats]
    x_positions = range(len(ordered_models))

    # Create two-line x-axis labels like in reference (e.g., "3.5\nHaiku")
    x_labels = []
    for m_id, m_disp in ordered_models:
        parts = m_disp.split(' ')
        if len(parts) == 2:
            x_labels.append(f"{parts[0]}\n{parts[1]}")
        else:
            x_labels.append(m_disp)

    for dim_id in dimensions:
        if dim_id not in DIMENSIONS:
            print(f"Warning: Unknown dimension {dim_id}")
            continue

        style = DIM_STYLES.get(dim_id, {'color': 'gray', 'linestyle': '-', 'marker': 'o'})

        means = []
        ses = []
        valid_x = []

        for i, (model_id, _) in enumerate(ordered_models):
            if dim_id in stats[model_id]['dims']:
                means.append(stats[model_id]['dims'][dim_id]['mean'])
                ses.append(stats[model_id]['dims'][dim_id]['se'])
                valid_x.append(i)

        if means:
            ax.errorbar(
                valid_x, means, yerr=ses,
                label=DIMENSIONS[dim_id],
                color=style['color'],
                linestyle=style['linestyle'],
                marker=style['marker'],
                markersize=7,
                linewidth=2,
                capsize=3,
                capthick=1,
            )

    # Styling
    ax.set_xlim(-0.5, len(ordered_models) - 0.5)
    ax.set_ylim(0, 100)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=11)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'], fontsize=11)

    # Box around plot
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)
        spine.set_color('black')

    ax.grid(True, axis='y', linestyle='-', alpha=0.3)

    # Legend inside the plot, upper right, in a box
    legend = ax.legend(
        loc='upper right',
        frameon=True,
        fontsize=10,
        edgecolor='black',
        fancybox=False,
        framealpha=1.0,
    )
    legend.get_frame().set_linewidth(1)

    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    print(f"Saved to {output_path} and {output_path.replace('.png', '.pdf')}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot Claude model evolution over time')
    parser.add_argument('--input', default='merged_final_results_with_claude45.parquet',
                        help='Input parquet file')
    parser.add_argument('--output', default='claude_evolution.png',
                        help='Output image path')
    parser.add_argument('--dimensions', nargs='+', default=DEFAULT_DIMS,
                        help='Dimensions to plot (default: ask_clarifying_questions maintain_social_boundaries)')
    parser.add_argument('--title', default=None,
                        help='Custom title for the plot')
    parser.add_argument('--all-dims', action='store_true',
                        help='Plot all 6 dimensions')
    args = parser.parse_args()

    if args.all_dims:
        dims = list(DIMENSIONS.keys())
    else:
        dims = args.dimensions

    print(f"Loading data from {args.input}...")
    stats = load_and_compute_stats(args.input)
    print(f"Found {len(stats)} Claude models")

    print(f"Plotting dimensions: {dims}")
    plot_claude_evolution(stats, dims, args.output, args.title)


if __name__ == '__main__':
    main()
