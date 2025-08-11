#!/usr/bin/env python3
"""
Krippendorff's Alpha Analysis - Individual Annotations Version
Compares 4 AI evaluators with individual human annotations (no averaging).
Each human annotation is matched directly to the corresponding AI evaluation.
"""

import pandas as pd
import numpy as np
import krippendorff
from pathlib import Path
from itertools import combinations
from scipy.stats import pearsonr
from rapidfuzz import fuzz
import warnings
warnings.filterwarnings('ignore')

def normalize_prompt(prompt):
    """Normalize prompt for matching using full-text hashing."""
    if pd.isna(prompt):
        return ""
    prompt = str(prompt)
    # Remove HTML tags
    import re
    prompt = re.sub(r'<[^>]+>', ' ', prompt)
    # Remove extra whitespace
    prompt = re.sub(r'\s+', ' ', prompt)
    # Convert to lowercase
    prompt = prompt.lower().strip()
    
    # Use full-text hashing to avoid collisions
    import hashlib
    return hashlib.sha256(prompt.encode()).hexdigest()

def fuzzy_match_dataframes(df1, df2, prompt_col='prompt', model_col='subject_model', 
                          score_col='score', threshold=95):
    """Match dataframes using fuzzy string matching for prompts.
    
    Args:
        df1: First dataframe (e.g., human annotations)
        df2: Second dataframe (e.g., evaluator data)
        prompt_col: Name of prompt column
        model_col: Name of model column
        score_col: Name of score column in df2
        threshold: Minimum similarity score (0-100) for matching
    
    Returns:
        Merged dataframe with matched rows
    """
    import re
    
    def clean_prompt(text):
        """Clean prompt for fuzzy matching."""
        if pd.isna(text):
            return ""
        text = str(text)
        # Replace <br> tags and newlines with space
        text = re.sub(r'<br\s*/?>', ' ', text)
        text = text.replace('\n', ' ').replace('\r', ' ')
        # Remove other HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.lower().strip()
    
    # First try exact matching after cleaning
    df1 = df1.copy()
    df2 = df2.copy()
    df1['prompt_clean'] = df1[prompt_col].apply(clean_prompt)
    df2['prompt_clean'] = df2[prompt_col].apply(clean_prompt)
    
    # Try exact match first (much faster)
    merged_exact = pd.merge(
        df1[[model_col, 'prompt_clean', score_col, prompt_col]],
        df2[[model_col, 'prompt_clean', score_col]],
        on=[model_col, 'prompt_clean'],
        suffixes=('', '_matched')
    )
    
    # If we got perfect coverage with exact matching, return it
    if len(merged_exact) == len(df1):
        return merged_exact
    
    # Otherwise, fall back to fuzzy matching for unmatched rows
    matched_models = set(zip(merged_exact[model_col], merged_exact['prompt_clean']))
    unmatched = df1[~df1.apply(lambda x: (x[model_col], x['prompt_clean']) in matched_models, axis=1)]
    
    matched_rows = merged_exact.to_dict('records')
    
    # Only do fuzzy matching for unmatched rows
    for _, row1 in unmatched.iterrows():
        model = row1[model_col]
        prompt1 = row1['prompt_clean']
        
        # Get all rows for this model from df2
        model_rows = df2[df2[model_col] == model]
        
        if len(model_rows) == 0:
            continue
        
        # Use first 200 chars for quick filtering
        prompt1_prefix = prompt1[:200]
        candidates = model_rows[model_rows['prompt_clean'].str[:200].apply(
            lambda x: fuzz.ratio(prompt1_prefix, x) > 80
        )]
        
        if len(candidates) == 0:
            candidates = model_rows
            
        # Find best match using fuzzy matching
        best_score = 0
        best_match = None
        
        for idx2, row2 in candidates.iterrows():
            prompt2 = row2['prompt_clean']
            # Use simple ratio for speed
            similarity = fuzz.ratio(prompt1[:500], prompt2[:500])
            
            if similarity > best_score:
                best_score = similarity
                best_match = row2
        
        # If best match exceeds threshold, add to results
        if best_score >= threshold and best_match is not None:
            matched_row = row1.to_dict()
            matched_row[f'{score_col}_matched'] = best_match[score_col]
            matched_row['match_score'] = best_score
            matched_rows.append(matched_row)
    
    return pd.DataFrame(matched_rows)

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
    
    # Define evaluator paths for direct comparison with human annotations
    evaluators = {
        'gemini': base_path / 'output' / 'openrouter_gemini_eval',
        'claude': base_path / 'output' / 'claude_4_sonnet_o3_continued_2',
        'o3': base_path / 'output' / '4_1_generated_o3_evaluated',
        '4.1': base_path / 'output' / '4_1_o3_continued'
    }
    
    # Load human annotations
    human_annotations_file = base_path / 'new_human_annotations.csv'
    
    print("=" * 80)
    print("KRIPPENDORFF'S ALPHA ANALYSIS - INDIVIDUAL ANNOTATIONS")
    print("=" * 80)
    
    if not human_annotations_file.exists():
        print(f"ERROR: {human_annotations_file} not found.")
        return
    
    # Load human annotations
    print("\nLoading data...")
    human_df = pd.read_csv(human_annotations_file)
    print(f"  Human annotations: {len(human_df):,} records")
    
    # Define tasks
    tasks = [
        'ask_clarifying_questions',
        'avoid_value_manipulation',
        'correct_misinformation',
        'defer_important_decisions',
        'encourage_learning',
        'maintain_social_boundaries'
    ]
    
    # Store all results
    all_ai_results = []
    all_human_results = []
    task_results = {}
    
    # Process each task
    print("\n" + "=" * 80)
    print("TASK-BY-TASK ANALYSIS")
    print("=" * 80)
    
    for task in tasks:
        print(f"\n{task.upper().replace('_', ' ')}")
        print("-" * 60)
        
        task_results[task] = {'ai_pairs': [], 'human': []}
        
        # Load evaluator data for this task
        evaluator_data = {}
        for name, path in evaluators.items():
            csv_path = path / task / 'raw.csv'
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                df['score'] = pd.to_numeric(df['score'], errors='coerce')
                df = df.dropna(subset=['score'])
                
                # Normalize prompt for deduplication
                df['prompt_norm'] = df['prompt'].apply(normalize_prompt)
                
                # Check for duplicates
                n_before = len(df)
                duplicates = df.duplicated(subset=['prompt_norm', 'subject_model'], keep=False)
                n_duplicates = duplicates.sum()
                
                # Remove duplicates - keep first occurrence
                df = df.drop_duplicates(subset=['prompt_norm', 'subject_model'], keep='first')
                n_after = len(df)
                
                evaluator_data[name] = df
                
                if n_duplicates > 0:
                    print(f"  {name}: {n_before:,} records → {n_after:,} (removed {n_before - n_after} duplicates)")
                else:
                    print(f"  {name}: {n_after:,} records")
        
        # Get human data for this task - NO GROUPING/AVERAGING
        task_human = human_df[human_df['dim'] == task].copy()
        if len(task_human) > 0:
            task_human['score'] = pd.to_numeric(task_human['score'], errors='coerce')
            task_human = task_human.dropna(subset=['score'])
            task_human['prompt_norm'] = task_human['prompt'].apply(normalize_prompt)
            
            # Count unique prompt-model pairs for reporting
            unique_pairs = task_human[['prompt_norm', 'subject_model']].drop_duplicates()
            print(f"  human: {len(task_human):,} individual annotations ({len(unique_pairs):,} unique prompt-model pairs)")
        
        # Calculate pairwise AI comparisons
        if len(evaluator_data) >= 2:
            print(f"\n  AI Evaluator Comparisons:")
            for eval1, eval2 in combinations(evaluator_data.keys(), 2):
                # Normalize prompts for matching
                df1 = evaluator_data[eval1].copy()
                df2 = evaluator_data[eval2].copy()
                df1['prompt_norm'] = df1['prompt'].apply(normalize_prompt)
                df2['prompt_norm'] = df2['prompt'].apply(normalize_prompt)
                
                # Merge on normalized prompt and subject_model
                merged = pd.merge(
                    df1[['prompt_norm', 'subject_model', 'score']],
                    df2[['prompt_norm', 'subject_model', 'score']],
                    on=['prompt_norm', 'subject_model'],
                    suffixes=('_1', '_2')
                )
                
                if len(merged) > 0:
                    scores1 = merged['score_1'].values
                    scores2 = merged['score_2'].values
                    
                    alpha = calculate_krippendorff_alpha(scores1, scores2)
                    correlation, p_value = pearsonr(scores1, scores2)
                    
                    result = {
                        'task': task,
                        'eval1': eval1,
                        'eval2': eval2,
                        'n_samples': len(scores1),
                        'alpha': alpha,
                        'correlation': correlation,
                        'p_value': p_value
                    }
                    
                    all_ai_results.append(result)
                    task_results[task]['ai_pairs'].append(result)
                    
                    print(f"    {eval1:8} vs {eval2:8}: α={alpha:6.3f}, r={correlation:6.3f}, n={len(scores1):,}")
        
        # Compare with human annotations - INDIVIDUAL ANNOTATIONS, NO AVERAGING
        if len(task_human) > 0:
            print(f"\n  Human Annotation Comparisons (Individual Annotations):")
            for eval_name, eval_df in evaluator_data.items():
                # Use fuzzy matching for human-evaluator comparison
                # But now we're matching individual annotations, not averaged ones
                matched = fuzzy_match_dataframes(
                    task_human,  # Using individual annotations directly
                    eval_df,
                    prompt_col='prompt',
                    model_col='subject_model',
                    score_col='score',
                    threshold=95  # High threshold for accuracy
                )
                
                if len(matched) > 0:
                    scores_eval = matched['score_matched'].values
                    scores_human = matched['score'].values
                    
                    alpha = calculate_krippendorff_alpha(scores_eval, scores_human)
                    correlation, p_value = pearsonr(scores_eval, scores_human)
                    
                    result = {
                        'task': task,
                        'evaluator': eval_name,
                        'n_samples': len(scores_eval),
                        'n_unique_pairs': len(matched[['prompt_clean', 'subject_model']].drop_duplicates()),
                        'alpha': alpha,
                        'correlation': correlation,
                        'p_value': p_value,
                        'coverage': len(matched) / len(task_human) * 100
                    }
                    
                    all_human_results.append(result)
                    task_results[task]['human'].append(result)
                    
                    print(f"    {eval_name:8} vs human: α={alpha:6.3f}, r={correlation:6.3f}, n={len(scores_eval):,} annotations ({result['coverage']:.1f}% coverage)")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    # Task-level summaries
    print("\n### Task-Level Agreement (Mean Values) ###")
    print(f"{'Task':<30} {'AI Agreement (α)':<20} {'Human Agreement (α)':<20}")
    print("-" * 70)
    
    for task in tasks:
        ai_alphas = [r['alpha'] for r in task_results[task]['ai_pairs'] if not np.isnan(r['alpha'])]
        human_alphas = [r['alpha'] for r in task_results[task]['human'] if not np.isnan(r['alpha'])]
        
        ai_mean = np.mean(ai_alphas) if ai_alphas else np.nan
        human_mean = np.mean(human_alphas) if human_alphas else np.nan
        
        task_display = task.replace('_', ' ').title()
        print(f"{task_display:<30} {ai_mean:<20.3f} {human_mean:<20.3f}")
    
    # Evaluator pair summaries
    print("\n### Inter-Evaluator Agreement (All Tasks Combined) ###")
    
    pair_stats = {}
    for result in all_ai_results:
        pair_key = f"{result['eval1']} vs {result['eval2']}"
        if pair_key not in pair_stats:
            pair_stats[pair_key] = {'alphas': [], 'correlations': [], 'n_samples': []}
        
        if not np.isnan(result['alpha']):
            pair_stats[pair_key]['alphas'].append(result['alpha'])
        pair_stats[pair_key]['correlations'].append(result['correlation'])
        pair_stats[pair_key]['n_samples'].append(result['n_samples'])
    
    print(f"\n{'Evaluator Pair':<20} {'Mean α':<12} {'Std α':<12} {'Mean r':<12} {'Total n':<12}")
    print("-" * 68)
    
    for pair, stats in sorted(pair_stats.items()):
        if stats['alphas']:
            mean_alpha = np.mean(stats['alphas'])
            std_alpha = np.std(stats['alphas'])
            mean_corr = np.mean(stats['correlations'])
            total_samples = sum(stats['n_samples'])
            
            print(f"{pair:<20} {mean_alpha:<12.3f} {std_alpha:<12.3f} {mean_corr:<12.3f} {total_samples:<12,}")
    
    # Human agreement summary
    print("\n### Human Agreement Summary (Individual Annotations) ###")
    
    evaluator_names = ['gemini', 'claude', 'o3', '4.1']
    print(f"\n{'Evaluator':<20} {'Mean α':<12} {'Std α':<12} {'Mean r':<12} {'Total Annotations':<20}")
    print("-" * 80)
    
    for eval_name in evaluator_names:
        eval_results = [r for r in all_human_results if r['evaluator'] == eval_name]
        if eval_results:
            eval_alphas = [r['alpha'] for r in eval_results if not np.isnan(r['alpha'])]
            eval_corrs = [r['correlation'] for r in eval_results]
            eval_samples = [r['n_samples'] for r in eval_results]
            
            if eval_alphas:
                print(f"{eval_name:<20} {np.mean(eval_alphas):<12.3f} {np.std(eval_alphas):<12.3f} "
                      f"{np.mean(eval_corrs):<12.3f} {sum(eval_samples):<20,}")
    
    # Overall statistics
    all_ai_alphas = [r['alpha'] for r in all_ai_results if not np.isnan(r['alpha'])]
    all_ai_correlations = [r['correlation'] for r in all_ai_results]
    all_human_alphas = [r['alpha'] for r in all_human_results if not np.isnan(r['alpha'])]
    all_human_correlations = [r['correlation'] for r in all_human_results]
    
    print("\n" + "=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)
    
    if all_ai_alphas:
        print("\n### AI Evaluators (All Tasks, All Pairs) ###")
        print(f"  Krippendorff's Alpha:")
        print(f"    Mean: {np.mean(all_ai_alphas):.3f}")
        print(f"    Std:  {np.std(all_ai_alphas):.3f}")
        print(f"    Min:  {np.min(all_ai_alphas):.3f}")
        print(f"    Max:  {np.max(all_ai_alphas):.3f}")
        
        print(f"\n  Pearson Correlation:")
        print(f"    Mean: {np.mean(all_ai_correlations):.3f}")
        print(f"    Std:  {np.std(all_ai_correlations):.3f}")
        print(f"    Min:  {np.min(all_ai_correlations):.3f}")
        print(f"    Max:  {np.max(all_ai_correlations):.3f}")
    
    if all_human_alphas:
        print("\n### Human Agreement (Individual Annotations) ###")
        print(f"  Krippendorff's Alpha:")
        print(f"    Mean: {np.mean(all_human_alphas):.3f}")
        print(f"    Std:  {np.std(all_human_alphas):.3f}")
        print(f"    Min:  {np.min(all_human_alphas):.3f}")
        print(f"    Max:  {np.max(all_human_alphas):.3f}")
        
        print(f"\n  Pearson Correlation:")
        print(f"    Mean: {np.mean(all_human_correlations):.3f}")
        print(f"    Std:  {np.std(all_human_correlations):.3f}")
        print(f"    Min:  {np.min(all_human_correlations):.3f}")
        print(f"    Max:  {np.max(all_human_correlations):.3f}")
    
    # Save detailed results to CSV
    if all_ai_results or all_human_results:
        # AI results
        ai_df = pd.DataFrame(all_ai_results)
        ai_df.to_csv(base_path / 'krippendorff_ai_individual_results.csv', index=False)
        
        # Human results
        human_results_df = pd.DataFrame(all_human_results)
        human_results_df.to_csv(base_path / 'krippendorff_human_individual_results.csv', index=False)
        
        print(f"\nDetailed results saved to:")
        print(f"  krippendorff_ai_individual_results.csv")
        print(f"  krippendorff_human_individual_results.csv")
    
    # Also save the matched individual annotations for inspection
    print("\nSaving matched individual annotations...")
    all_matched_data = []
    
    for task in tasks:
        task_human = human_df[human_df['dim'] == task].copy()
        if len(task_human) > 0:
            task_human['score'] = pd.to_numeric(task_human['score'], errors='coerce')
            task_human = task_human.dropna(subset=['score'])
            
            for name, path in evaluators.items():
                csv_path = path / task / 'raw.csv'
                if csv_path.exists():
                    df = pd.read_csv(csv_path)
                    df['score'] = pd.to_numeric(df['score'], errors='coerce')
                    df = df.dropna(subset=['score'])
                    
                    # Match with fuzzy matching
                    matched = fuzzy_match_dataframes(
                        task_human,
                        df,
                        prompt_col='prompt',
                        model_col='subject_model',
                        score_col='score',
                        threshold=95
                    )
                    
                    if len(matched) > 0:
                        matched['task'] = task
                        matched['evaluator'] = name
                        matched['human_score'] = matched['score']
                        matched['ai_score'] = matched['score_matched']
                        
                        # Keep essential columns
                        matched = matched[['task', 'evaluator', 'subject_model', 'prompt', 
                                         'human_score', 'ai_score']]
                        all_matched_data.append(matched)
    
    if all_matched_data:
        matched_df = pd.concat(all_matched_data, ignore_index=True)
        matched_file = base_path / 'krippendorff_matched_individual_annotations.csv'
        matched_df.to_csv(matched_file, index=False)
        print(f"  Matched individual annotations saved to: {matched_file.name}")
        print(f"  Total matched annotations: {len(matched_df):,}")

if __name__ == "__main__":
    import sys
    
    # Simple output capture to text file
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
    txt_file = Path(__file__).parent / 'krippendorff_individual_analysis_report.txt'
    with open(txt_file, 'w') as f:
        original_stdout = sys.stdout
        sys.stdout = Tee(sys.stdout, f)
        main()
        sys.stdout = original_stdout
    
    print(f"\nFull report also saved to: {txt_file.name}")