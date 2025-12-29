#!/usr/bin/env python3
"""
Export evaluation results in standard formats for visualization and sharing.

Outputs:
1. model_results_summary.csv - Per-dimension means and SEs in visualization format
2. Updates the merged parquet with all raw scores
"""

import pandas as pd
import numpy as np
import argparse
import os

# Model to provider mapping
MODEL_PROVIDERS = {
    # Anthropic
    'claude-3-haiku-20240307': ('Claude 3 Haiku', 'Anthropic'),
    'claude-3-sonnet-20240229': ('Claude 3 Sonnet', 'Anthropic'),
    'claude-3-opus-20240229': ('Claude 3 Opus', 'Anthropic'),
    'claude-3-5-haiku-20241022': ('Claude 3.5 Haiku', 'Anthropic'),
    'claude-3-5-sonnet-20240620': ('Claude 3.5 Sonnet', 'Anthropic'),
    'claude-3-5-sonnet-20241022': ('Claude 3.5 Sonnet (Oct)', 'Anthropic'),
    'claude-3-7-sonnet-20250219': ('Claude 3.7 Sonnet', 'Anthropic'),
    'claude-sonnet-4-20250514': ('Claude 4 Sonnet', 'Anthropic'),
    'claude-opus-4.1-20250805': ('Claude 4.1 Opus', 'Anthropic'),
    'claude-sonnet-4-5-20250929': ('Claude 4.5 Sonnet', 'Anthropic'),
    'claude-opus-4-5-20251101': ('Claude 4.5 Opus', 'Anthropic'),

    # OpenAI
    'gpt-4o': ('GPT-4o', 'OpenAI'),
    'gpt-4o-mini': ('GPT-4o Mini', 'OpenAI'),
    'gpt-4.1': ('GPT-4.1', 'OpenAI'),
    'gpt-4.1-mini': ('GPT-4.1 Mini', 'OpenAI'),
    'gpt-5': ('GPT-5', 'OpenAI'),
    'gpt-5-high': ('GPT-5 High', 'OpenAI'),
    'o3-mini-2025-01-31': ('O3 Mini', 'OpenAI'),
    'o3-2025-04-16': ('O3', 'OpenAI'),
    'o4-mini-2025-04-16': ('O4 Mini', 'OpenAI'),

    # Google
    'gemini-1.5-flash': ('Gemini 1.5 Flash', 'Google'),
    'gemini-2.0-flash': ('Gemini 2.0 Flash', 'Google'),
    'gemini-2.5-flash-preview-04-17': ('Gemini 2.5 Flash', 'Google'),
    'gemini-2.5-pro-preview-03-25': ('Gemini 2.5 Pro', 'Google'),

    # Meta
    'meta-llama-3-70b-instruct': ('Llama 3 70B', 'Meta'),
    'llama-4-scout-instruct': ('Llama 4 Scout', 'Meta'),
    'llama-4-maverick-instruct': ('Llama 4 Maverick', 'Meta'),

    # xAI
    'grok-3': ('Grok 3', 'xAI'),
    'x-ai/grok-4': ('Grok 4', 'xAI'),
}

DIMENSION_DISPLAY = {
    'ask_clarifying_questions': 'Ask Clarifying Questions',
    'avoid_value_manipulation': 'Avoid Value Manipulation',
    'correct_misinformation': 'Correct Misinformation',
    'defer_important_decisions': 'Defer Important Decisions',
    'encourage_learning': 'Encourage Learning',
    'maintain_social_boundaries': 'Maintain Social Boundaries',
}


def export_summary_csv(df: pd.DataFrame, output_path: str):
    """Export results in visualization format: Model, Provider, Dim, Mean (%), SE (%)"""

    rows = []

    # Calculate overall stats first (the "-" dimension row)
    overall_means = df.groupby('subject_model')['score'].mean() * 10  # Convert to %
    overall_stds = df.groupby('subject_model')['score'].std()
    overall_counts = df.groupby('subject_model')['score'].count()
    overall_ses = (overall_stds / np.sqrt(overall_counts)) * 10  # SE as %

    # Grand overall (all models combined)
    grand_mean = df['score'].mean() * 10
    grand_se = (df['score'].std() / np.sqrt(len(df))) * 10
    rows.append({
        'Model': '-',
        'Provider': '-',
        'Dim': '-',
        'Mean (%)': round(grand_mean, 2),
        'SE (%)': round(grand_se, 2)
    })

    # Per-model overall
    for model_id in df['subject_model'].unique():
        display_name, provider = MODEL_PROVIDERS.get(model_id, (model_id, 'Unknown'))
        rows.append({
            'Model': display_name,
            'Provider': provider,
            'Dim': '-',
            'Mean (%)': round(overall_means[model_id], 2),
            'SE (%)': round(overall_ses[model_id], 2)
        })

    # Per-model per-dimension
    for model_id in df['subject_model'].unique():
        display_name, provider = MODEL_PROVIDERS.get(model_id, (model_id, 'Unknown'))

        for dim_id, dim_display in DIMENSION_DISPLAY.items():
            subset = df[(df['subject_model'] == model_id) & (df['dimension'] == dim_id)]
            if len(subset) == 0:
                continue

            mean_score = subset['score'].mean() * 10
            se_score = (subset['score'].std() / np.sqrt(len(subset))) * 10

            rows.append({
                'Model': display_name,
                'Provider': provider,
                'Dim': dim_display,
                'Mean (%)': round(mean_score, 2),
                'SE (%)': round(se_score, 2)
            })

    result_df = pd.DataFrame(rows)
    result_df.to_csv(output_path, index=False)
    print(f"Saved summary to {output_path}")
    return result_df


def main():
    parser = argparse.ArgumentParser(description='Export evaluation results')
    parser.add_argument('--input', default='merged_final_results_with_claude45.parquet',
                        help='Input parquet file with all results')
    parser.add_argument('--output-summary', default='model_results_summary.csv',
                        help='Output CSV for visualization summary')
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df)} rows, {df['subject_model'].nunique()} models")

    export_summary_csv(df, args.output_summary)

    print("\nDone!")


if __name__ == '__main__':
    main()
