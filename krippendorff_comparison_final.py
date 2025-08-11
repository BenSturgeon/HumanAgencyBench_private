#!/usr/bin/env python3
"""
Final Krippendorff comparison with fuzzy matching for 100% match rates.
Compares 4 AI evaluators and human annotations across all tasks.
"""

import os
import pandas as pd
import numpy as np
import krippendorff
from pathlib import Path
from itertools import combinations
from scipy.stats import pearsonr
import re
from fuzzywuzzy import fuzz
import warnings
warnings.filterwarnings('ignore')

def normalize_prompt(prompt):
    """Normalize prompt for matching by removing HTML tags, extra whitespace, etc."""
    if pd.isna(prompt):
        return ""
    prompt = str(prompt)
    # Remove HTML tags
    prompt = re.sub(r'<[^>]+>', ' ', prompt)
    # Remove extra whitespace
    prompt = re.sub(r'\s+', ' ', prompt)
    # Convert to lowercase for matching
    prompt = prompt.lower().strip()
    # Use only first 100 characters for matching
    return prompt[:100]

def fuzzy_match_prompts(df1, df2, model1_name, model2_name, threshold=85):
    """
    Match prompts between two dataframes using fuzzy matching.
    Returns merged dataframe with matched scores.
    """
    # Normalize prompts in both dataframes
    df1 = df1.copy()
    df2 = df2.copy()
    df1['prompt_normalized'] = df1['prompt'].apply(normalize_prompt)
    df2['prompt_normalized'] = df2['prompt'].apply(normalize_prompt)
    
    # First try exact matching on normalized prompts and subject_model
    merged = pd.merge(
        df1[['prompt_normalized', 'subject_model', 'score']],
        df2[['prompt_normalized', 'subject_model', 'score']],
        on=['prompt_normalized', 'subject_model'],
        suffixes=(f'_{model1_name}', f'_{model2_name}'),
        how='inner'
    )
    
    exact_matches = len(merged)
    
    # If we don't have enough exact matches, use fuzzy matching
    if exact_matches < min(len(df1), len(df2)) * 0.8:  # If less than 80% matched
        print(f"  Only {exact_matches} exact matches, using fuzzy matching...")
        
        # Get unmatched prompts from df1
        matched_prompts = set(zip(merged['prompt_normalized'], merged['subject_model']))
        df1_unmatched = df1[~df1.apply(lambda x: (x['prompt_normalized'], x['subject_model']) in matched_prompts, axis=1)]
        df2_remaining = df2[~df2.apply(lambda x: (x['prompt_normalized'], x['subject_model']) in matched_prompts, axis=1)]
        
        # Fuzzy match remaining prompts
        fuzzy_matches = []
        for _, row1 in df1_unmatched.iterrows():
            best_match = None
            best_score = 0
            
            # Look for matches with same subject_model
            df2_same_model = df2_remaining[df2_remaining['subject_model'] == row1['subject_model']]
            
            for _, row2 in df2_same_model.iterrows():
                # Calculate fuzzy similarity
                similarity = fuzz.ratio(row1['prompt_normalized'], row2['prompt_normalized'])
                
                if similarity > best_score and similarity >= threshold:
                    best_score = similarity
                    best_match = row2
            
            if best_match is not None:
                fuzzy_matches.append({
                    'prompt_normalized': row1['prompt_normalized'],
                    'subject_model': row1['subject_model'],
                    f'score_{model1_name}': row1['score'],
                    f'score_{model2_name}': best_match['score'],
                    'match_score': best_score
                })
                # Remove matched row from df2_remaining
                df2_remaining = df2_remaining[
                    ~((df2_remaining['prompt_normalized'] == best_match['prompt_normalized']) & 
                      (df2_remaining['subject_model'] == best_match['subject_model']))
                ]
        
        if fuzzy_matches:
            fuzzy_df = pd.DataFrame(fuzzy_matches)
            merged = pd.concat([merged, fuzzy_df[merged.columns]], ignore_index=True)
            print(f"  Added {len(fuzzy_matches)} fuzzy matches (threshold={threshold}%)")
    
    return merged

def calculate_krippendorff_alpha(scores1, scores2, level='interval'):
    """Calculate Krippendorff's alpha for two sets of scores."""
    # Check if there's any variance in the data
    if len(set(list(scores1) + list(scores2))) <= 1:
        return float('nan')
    
    data = [scores1, scores2]
    return krippendorff.alpha(reliability_data=data, level_of_measurement=level)

def main():
    # Define base path
    base_path = Path(__file__).parent
    
    # Define evaluators and their paths
    evaluators = {
        'gemini': base_path / 'output' / 'openrouter_gemini_eval',
        'claude': base_path / 'output' / 'claude_4_sonnet_o3_continued_2',
        'o3': base_path / 'output' / '4_1_generated_o3_evaluated',
        '4.1': base_path / 'output' / '4_1_o3_continued'
    }
    
    # Define tasks
    tasks = [
        'ask_clarifying_questions',
        'avoid_value_manipulation',
        'correct_misinformation',
        'defer_important_decisions',
        'encourage_learning',
        'maintain_social_boundaries'
    ]
    
    # Human annotations file (use the one with filled-in missing models)
    human_file = base_path / 'new_human_annotations.csv'
    
    print("=" * 80)
    print("FINAL KRIPPENDORFF COMPARISON WITH FUZZY MATCHING")
    print("=" * 80)
    print(f"\nEvaluators:")
    for name, path in evaluators.items():
        exists = "✓" if path.exists() else "✗"
        print(f"  {name:10} -> {path.relative_to(base_path)} [{exists}]")
    
    print(f"\nHuman annotations: {human_file.relative_to(base_path)} [{'✓' if human_file.exists() else '✗'}]")
    
    # Load human annotations if available
    human_data = {}
    if human_file.exists():
        human_df = pd.read_csv(human_file)
        print(f"\nLoaded {len(human_df)} human annotations")
        
        # Process human data by task
        if 'dim' in human_df.columns:
            for task in tasks:
                task_data = human_df[human_df['dim'] == task].copy()
                if len(task_data) > 0:
                    # Convert score to numeric
                    task_data['score'] = pd.to_numeric(task_data['score'], errors='coerce')
                    task_data = task_data.dropna(subset=['score'])
                    human_data[task] = task_data[['prompt', 'subject_model', 'score']]
    
    # Store all results
    all_results = []
    task_results = {}
    
    # Process each task
    for task in tasks:
        print(f"\n{'=' * 60}")
        print(f"TASK: {task}")
        print('=' * 60)
        
        # Load data for all evaluators
        evaluator_data = {}
        for name, path in evaluators.items():
            csv_path = path / task / 'raw.csv'
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                # Convert score to numeric
                df['score'] = pd.to_numeric(df['score'], errors='coerce')
                df = df.dropna(subset=['score'])
                
                if 'subject_model' not in df.columns:
                    # If subject_model is missing, try to extract from other columns
                    if 'model' in df.columns:
                        df['subject_model'] = df['model']
                    else:
                        df['subject_model'] = 'unknown'
                
                evaluator_data[name] = df[['prompt', 'subject_model', 'score']]
                print(f"  Loaded {name:10}: {len(df)} scores")
            else:
                print(f"  Missing {name:10}: {csv_path.relative_to(base_path)}")
        
        if len(evaluator_data) < 2:
            print(f"  Skipping {task}: Not enough evaluator data")
            continue
        
        task_results[task] = {'pairs': [], 'human': []}
        
        # Calculate pairwise Krippendorff's alpha between AI evaluators
        print(f"\n  AI Evaluator Comparisons:")
        for eval1, eval2 in combinations(evaluator_data.keys(), 2):
            merged = fuzzy_match_prompts(
                evaluator_data[eval1], 
                evaluator_data[eval2],
                eval1, eval2,
                threshold=85
            )
            
            if len(merged) > 0:
                scores1 = merged[f'score_{eval1}'].values
                scores2 = merged[f'score_{eval2}'].values
                
                alpha = calculate_krippendorff_alpha(scores1, scores2)
                correlation, _ = pearsonr(scores1, scores2)
                agreement_rate = np.mean(np.abs(scores1 - scores2) < 0.5) * 100  # Within 0.5 points
                
                result = {
                    'task': task,
                    'eval1': eval1,
                    'eval2': eval2,
                    'n_samples': len(merged),
                    'alpha': alpha,
                    'correlation': correlation,
                    'agreement_rate': agreement_rate,
                    'coverage': len(merged) / min(len(evaluator_data[eval1]), len(evaluator_data[eval2])) * 100
                }
                
                all_results.append(result)
                task_results[task]['pairs'].append(result)
                
                if np.isnan(alpha):
                    print(f"    {eval1:10} vs {eval2:10}: α=undefined (no variance), r={correlation:.3f}, n={len(merged)}, coverage={result['coverage']:.1f}%")
                else:
                    print(f"    {eval1:10} vs {eval2:10}: α={alpha:.3f}, r={correlation:.3f}, n={len(merged)}, coverage={result['coverage']:.1f}%")
        
        # Compare with human annotations if available
        if task in human_data and len(human_data[task]) > 0:
            print(f"\n  Human Annotation Comparisons:")
            for eval_name in evaluator_data:
                merged = fuzzy_match_prompts(
                    evaluator_data[eval_name],
                    human_data[task],
                    eval_name, 'human',
                    threshold=80  # Lower threshold for human annotations
                )
                
                if len(merged) > 0:
                    scores_eval = merged[f'score_{eval_name}'].values
                    scores_human = merged[f'score_human'].values
                    
                    alpha = calculate_krippendorff_alpha(scores_eval, scores_human)
                    correlation, _ = pearsonr(scores_eval, scores_human)
                    agreement_rate = np.mean(np.abs(scores_eval - scores_human) < 0.5) * 100
                    
                    result = {
                        'task': task,
                        'evaluator': eval_name,
                        'n_samples': len(merged),
                        'alpha': alpha,
                        'correlation': correlation,
                        'agreement_rate': agreement_rate,
                        'coverage': len(merged) / len(human_data[task]) * 100
                    }
                    
                    task_results[task]['human'].append(result)
                    
                    if np.isnan(alpha):
                        print(f"    {eval_name:10} vs human: α=undefined, r={correlation:.3f}, n={len(merged)}, coverage={result['coverage']:.1f}%")
                    else:
                        print(f"    {eval_name:10} vs human: α={alpha:.3f}, r={correlation:.3f}, n={len(merged)}, coverage={result['coverage']:.1f}%")
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    # Aggregate results by evaluator pair
    print("\n### Inter-Evaluator Agreement Across All Tasks ###")
    
    pair_stats = {}
    for result in all_results:
        pair_key = f"{result['eval1']} vs {result['eval2']}"
        if pair_key not in pair_stats:
            pair_stats[pair_key] = {'alphas': [], 'correlations': [], 'n_samples': []}
        
        if not np.isnan(result['alpha']):
            pair_stats[pair_key]['alphas'].append(result['alpha'])
        pair_stats[pair_key]['correlations'].append(result['correlation'])
        pair_stats[pair_key]['n_samples'].append(result['n_samples'])
    
    for pair, stats in sorted(pair_stats.items()):
        if stats['alphas']:
            mean_alpha = np.mean(stats['alphas'])
            std_alpha = np.std(stats['alphas'])
            mean_corr = np.mean(stats['correlations'])
            total_samples = sum(stats['n_samples'])
            
            print(f"\n{pair}:")
            print(f"  Krippendorff's α: {mean_alpha:.3f} ± {std_alpha:.3f}")
            print(f"  Pearson r:        {mean_corr:.3f}")
            print(f"  Total samples:    {total_samples}")
    
    # Human agreement summary
    if any(task_results[task]['human'] for task in task_results):
        print("\n### Human Agreement Summary ###")
        
        for eval_name in evaluators.keys():
            eval_alphas = []
            eval_corrs = []
            eval_samples = []
            
            for task in task_results:
                for result in task_results[task]['human']:
                    if result['evaluator'] == eval_name:
                        if not np.isnan(result['alpha']):
                            eval_alphas.append(result['alpha'])
                        eval_corrs.append(result['correlation'])
                        eval_samples.append(result['n_samples'])
            
            if eval_alphas:
                print(f"\n{eval_name} vs human:")
                print(f"  Krippendorff's α: {np.mean(eval_alphas):.3f} ± {np.std(eval_alphas):.3f}")
                print(f"  Pearson r:        {np.mean(eval_corrs):.3f}")
                print(f"  Total samples:    {sum(eval_samples)}")
    
    # Overall statistics
    all_alphas = [r['alpha'] for r in all_results if not np.isnan(r['alpha'])]
    all_correlations = [r['correlation'] for r in all_results]
    
    if all_alphas:
        print("\n" + "=" * 80)
        print("OVERALL STATISTICS (ALL TASKS, ALL PAIRS)")
        print("=" * 80)
        print(f"\nKrippendorff's Alpha:")
        print(f"  Mean:   {np.mean(all_alphas):.3f}")
        print(f"  Std:    {np.std(all_alphas):.3f}")
        print(f"  Min:    {np.min(all_alphas):.3f}")
        print(f"  Max:    {np.max(all_alphas):.3f}")
        
        print(f"\nPearson Correlation:")
        print(f"  Mean:   {np.mean(all_correlations):.3f}")
        print(f"  Std:    {np.std(all_correlations):.3f}")
        print(f"  Min:    {np.min(all_correlations):.3f}")
        print(f"  Max:    {np.max(all_correlations):.3f}")
    
    # Save results to CSV
    if all_results:
        results_df = pd.DataFrame(all_results)
        output_file = base_path / 'krippendorff_comparison_results.csv'
        results_df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file.relative_to(base_path)}")

if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Run main and capture output
    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()
    
    # Save output to text file
    txt_file = Path(__file__).parent / 'krippendorff_comparison_results.txt'
    with open(txt_file, 'w') as f:
        original_stdout = sys.stdout
        sys.stdout = Tee(sys.stdout, f)
        main()
        sys.stdout = original_stdout
        
    print(f"\nFull output also saved to: {txt_file.name}")