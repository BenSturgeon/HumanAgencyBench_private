#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotly import graph_objects as go
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import os

# %% Load and concat all data
dfs = []

# Check output directory for data
for directory in ['output']:
    for subdir in os.listdir(directory):
        # Skip non-directories and special cases
        if not os.path.isdir(f'{directory}/{subdir}'):
            continue

        # Look for CSV files
        csv_path = None
        if 'raw.csv' in os.listdir(f'{directory}/{subdir}'):
            csv_path = f'{directory}/{subdir}/raw.csv'
        elif any(file.endswith('.csv') for file in os.listdir(f'{directory}/{subdir}')):
            # Take the first CSV file found
            csv_files = [file for file in os.listdir(f'{directory}/{subdir}') if file.endswith('.csv')]
            csv_path = f'{directory}/{subdir}/{csv_files[0]}'

        if csv_path:
            df = pd.read_csv(csv_path)
            df['subdim'] = subdir
            df['source_dir'] = directory
            dfs.append(df)

df = pd.concat(dfs)

# %% Filter relevant subdimensions
relevant_subdims = [
    'acknowledge_limitations',
    'ask_clarifying_questions',
    'avoid_value_manipulation',
    'correct_misinformation', 
    'defer_important_decisions',
    'encourage_learning',
    'follow_instructions',
    'maintain_social_boundaries'
]
df = df[df['subdim'].isin(relevant_subdims)]

# %% Normalize scores
def normalize_scores(df):
    subdim = df['subdim'].iloc[0]
    if subdim == "correct_misinfo_llama_claude" or subdim == "correct_misinformation_claude_qa":
        subdim = "correct_misinformation"

    if subdim == "asks_follow_up_question_claude" or subdim == "asks_follow_up_question_claude_eval":
        subdim = "asks_follow_up_question"

    max_score = 10
    df['score'] = df['score'] / max_score

    return df

df = df.groupby('subdim').apply(normalize_scores).reset_index(drop=True)

# %% Save raw concatenated data
df.to_csv('concatenated_raw.csv', index=False)

# %% Clean up model and subdim names
df['subject_model'] = df['subject_model'].replace({
    'gpt-4o': 'GPT-4o',
    'gpt-4o-mini': 'GPT-4o Mini',
    'o1-mini': 'O1 Mini',
    'models/gemini-1.5-flash-latest': 'Gemini Flash',
    'models/gemini-1.5-pro-latest': 'Gemini Pro',
    'gemma2-9b-it': 'Gemma2 9B',
    'claude-3-5-sonnet-20240620': 'Claude 3.5 Sonnet',
    'claude-3-5-sonnet-20241022': 'Claude 3.5 Sonnet (New)',
    'claude-3-7-sonnet-20250219': 'Claude 3.7 Sonnet',
    'claude-3-sonnet-20240229': 'Claude 3 Sonnet',
    'llama-3.1-8b-instant': 'Llama 3.1 8B',
    'llama-3.2-3b-preview': 'Llama 3.2 3B',
    'meta/meta-llama-3.1-405b-instruct': 'Llama 3.1 405B',
    'mixtral-8x7b-32768': 'Mixtral 8x7B',
})

df['subdim'] = df['subdim'].replace({
    'acknowledge_limitations': 'Acknowledge Limitations',
    'ask_clarifying_questions': 'Ask Clarifying Questions', 
    'avoid_value_manipulation': 'Avoid Value Manipulation',
    'correct_misinformation': 'Correct Misinformation',
    'defer_important_decisions': 'Defer Important Decisions',
    'encourage_learning': 'Encourage Learning',
    'follow_instructions': 'Follow Instructions',
    'maintain_social_boundaries': 'Maintain Social Boundaries'
})


# %% Create pivot table
pivot_table = df.pivot_table(
    values='score',
    index='subdim',
    columns='subject_model',
    aggfunc='mean'
)

# %% Add subdim average performance
average_row = pivot_table.mean(axis=0)
pivot_table.loc['Mean Score'] = average_row

# %% Create visualization
# Set the style to dark background
plt.style.use('dark_background')

# Define colors
dark_gray = (52/255, 55/255, 67/255)      # #343743
light_green = (159/255, 252/255, 189/255)  # #9FFCBD

# Define text sizes
TITLE_SIZE = 14
LABEL_SIZE = 13
TICK_SIZE = 12
PROVIDER_SIZE = 18

# Define provider groups
provider_groups = {
    "OpenAI": ['GPT-4o', 'GPT-4o Mini', 'O1 Mini'],
    "Anthropic": ['Claude 3.7 Sonnet', 'Claude 3.5 Sonnet (New)', 'Claude 3.5 Sonnet'],
    "Google": ['Gemini Flash', 'Gemini Pro'],
    "Meta": ['Llama 3.2 3B', 'Llama 3.1 405B', 'Llama 3.1 8B']
}

# Replace NaN and 0 values with 0.001
pivot_table = pivot_table.fillna(0.001)
pivot_table = pivot_table.replace(0, 0.001)

# Create the base colormap
colors = [dark_gray, light_green]  # Dark to light
n_bins = 256
base_cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)

# %% Define helper function for gradient bars
def create_gradient_bar(ax, x, y, width, height):
    """Create a horizontal bar with a gradient normalized to the x-axis range."""
    # Handle zero or very small values
    if width < 0.001:
        rect = Rectangle((x, y-height/2), 0.001, height, color=dark_gray, alpha=0.5)
        ax.add_patch(rect)
        return None

    gradient_x = np.linspace(0, width, max(int(width * n_bins), 2))
    gradient_colors = base_cmap(gradient_x)
    gradient = gradient_colors.reshape(1, -1, 4)

    im = ax.imshow(gradient, aspect='auto', extent=[x, x+width, y-height/2, y+height/2])
    rect = Rectangle((x, y-height/2), width, height, fill=False, color='white', linewidth=0.5)
    ax.add_patch(rect)
    return im

# %% Create and save visualization
subdims = [dim for dim in pivot_table.index if dim != 'Mean Score']

fig, axs = plt.subplots(len(provider_groups), len(subdims), 
                      figsize=(20, 12),  
                      gridspec_kw={'hspace': 0.4, 'wspace': 0.3})

fig.patch.set_facecolor(dark_gray)

for i, (provider, models) in enumerate(provider_groups.items()):
    for j, subdim in enumerate(subdims):
        ax = axs[i, j]
        ax.set_facecolor(dark_gray)
        
        # Filter to only include models that exist in the pivot table
        available_models = [model for model in models if model in pivot_table.columns]
        
        if available_models:
            scores = pivot_table.loc[subdim, available_models]

            # Create gradient bars with increased height
            for idx, (model, score) in enumerate(scores.items()):
                create_gradient_bar(ax, 0, idx, score, 0.9)

            # Customize the subplot
            ax.set_xlim(-0.03, 1.01)
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_xticklabels(['0', '0.25', '0.5', '0.75', '1'], rotation=45, color='white', fontsize=TICK_SIZE)
            ax.grid(True, axis='x', linestyle='--', alpha=0.3, color='white')

            # Set y-axis limits and ticks with more padding
            ax.set_ylim(-0.7, len(available_models) - 0.3)

            # Only show y-labels (model names) for the first column
            if j == 0:
                ax.set_yticks(np.arange(len(available_models)))
                ax.set_yticklabels(available_models, color='white', fontsize=LABEL_SIZE)
                ax.set_ylabel(provider, fontweight='bold', labelpad=20, fontsize=PROVIDER_SIZE, color='white')
            else:
                ax.set_yticks([])

            # Only show dimension names for the top row
            if i == 0:
                ax.set_title(subdim, pad=10, rotation=15, color='white', fontsize=TITLE_SIZE)

            # Set spine colors to white
            for spine in ax.spines.values():
                spine.set_color('white')

plt.tight_layout()
plt.subplots_adjust(left=0.15)
plt.savefig('model_comparison.png', dpi=700, bbox_inches='tight', facecolor=dark_gray)
plt.show()

# %% Display additional information
print(f"Number of dimensions in pivot table: {len(pivot_table.index)}")
print("Unique model names in DataFrame:")
print(pivot_table.columns.tolist())

