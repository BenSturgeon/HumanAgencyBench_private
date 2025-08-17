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
    import hashlib
    
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
        # Normalize apostrophes and quotes
        text = text.replace(''', "'").replace(''', "'")  # Smart quotes to regular
        text = text.replace('"', '"').replace('"', '"')  # Smart double quotes
        text = text.replace('–', '-').replace('—', '-')  # Em and en dashes
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.lower().strip()
    
    def generate_prompt_id(prompt, model):
        """Generate a unique ID for a prompt-model pair."""
        combined = f"{prompt}_{model}"
        return hashlib.md5(combined.encode()).hexdigest()[:16]
    
    # First try exact matching after cleaning
    df1 = df1.copy()
    df2 = df2.copy()
    df1['prompt_clean'] = df1[prompt_col].apply(clean_prompt)
    df2['prompt_clean'] = df2[prompt_col].apply(clean_prompt)
    
    # Add truncated prompts for easier inspection
    df1['prompt_truncated'] = df1['prompt_clean'].str[:100]
    df2['prompt_truncated'] = df2['prompt_clean'].str[:100]
    
    # Generate prompt IDs
    df1['prompt_id'] = df1.apply(lambda x: generate_prompt_id(x['prompt_clean'], x[model_col]), axis=1)
    df2['prompt_id'] = df2.apply(lambda x: generate_prompt_id(x['prompt_clean'], x[model_col]), axis=1)
    
    # Try exact match first (much faster)
    merged_exact = pd.merge(
        df1[[model_col, 'prompt_clean', score_col, prompt_col, 'prompt_truncated', 'prompt_id']],
        df2[[model_col, 'prompt_clean', score_col, 'prompt_truncated', 'prompt_id']],
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
            # Ensure we have prompt_id and prompt_truncated
            if 'prompt_id' not in matched_row:
                matched_row['prompt_id'] = generate_prompt_id(row1['prompt_clean'], row1[model_col])
            if 'prompt_truncated' not in matched_row:
                matched_row['prompt_truncated'] = row1['prompt_clean'][:100]
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
    print("GENERATING REPLICATION DATASET")
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
    
    # Create one giant CSV with all cleaned data from humans and all 4 evaluators
    print("\nCreating unified dataset with all annotations...")
    all_annotations = []
    
    # Helper function for cleaning prompts
    import re
    import hashlib
    
    def clean_prompt_for_matching(text):
        if pd.isna(text):
            return ""
        text = str(text)
        text = re.sub(r'<br\s*/?>', ' ', text)
        text = text.replace('\n', ' ').replace('\r', ' ')
        text = re.sub(r'<[^>]+>', ' ', text)
        # Normalize apostrophes and quotes
        text = text.replace(''', "'").replace(''', "'")  # Smart quotes to regular
        text = text.replace('"', '"').replace('"', '"')  # Smart double quotes
        text = text.replace('–', '-').replace('—', '-')  # Em and en dashes
        text = re.sub(r'\s+', ' ', text)
        return text.lower().strip()
    
    def generate_prompt_id(prompt, model):
        combined = f"{prompt}_{model}"
        return hashlib.md5(combined.encode()).hexdigest()[:16]
    
    # For fuzzy matching when exact match fails
    def find_fuzzy_match_id(prompt_clean, model, existing_data, threshold=99):
        """Find a matching prompt_id using fuzzy matching."""
        from rapidfuzz import fuzz
        best_match_id = None
        best_score = 0
        
        # Look for similar prompts with same model
        candidates = existing_data[existing_data['subject_model'] == model]
        for _, row in candidates.iterrows():
            similarity = fuzz.ratio(prompt_clean, row.get('prompt_clean', ''))
            if similarity > best_score and similarity >= threshold:
                best_score = similarity
                best_match_id = row.get('prompt_id')
        
        return best_match_id if best_score >= threshold else None
    
    # First pass - collect all evaluator data to enable fuzzy matching
    evaluator_annotations = []
    
    for task in tasks:
        print(f"  Processing {task}...")
        
        # Process each evaluator's data first
        for eval_name, eval_path in evaluators.items():
            csv_path = eval_path / task / 'raw.csv'
            if csv_path.exists():
                eval_df = pd.read_csv(csv_path)
                eval_df['score'] = pd.to_numeric(eval_df['score'], errors='coerce')
                eval_df = eval_df.dropna(subset=['score'])
                eval_df['prompt_clean'] = eval_df['prompt'].apply(clean_prompt_for_matching)
                eval_df['prompt_truncated'] = eval_df['prompt_clean'].str[:100]
                eval_df['prompt_id'] = eval_df.apply(
                    lambda x: generate_prompt_id(x['prompt_clean'], x['subject_model']), axis=1
                )
                
                # Remove duplicates
                eval_df = eval_df.drop_duplicates(subset=['prompt_id', 'subject_model'], keep='first')
                
                # Add evaluator annotations to list
                for _, row in eval_df.iterrows():
                    evaluator_annotations.append({
                        'task': task,
                        'evaluator': eval_name,
                        'annotator_id': f'{eval_name}_evaluator',
                        'subject_model': row['subject_model'],
                        'prompt_id': row['prompt_id'],
                        'prompt_truncated': row['prompt_truncated'],
                        'prompt_clean': row['prompt_clean'],
                        'prompt_original': row['prompt'],
                        'subject_response': row.get('subject_response', row.get('response', '')),
                        'score': row['score']
                    })
    
    # Convert to DataFrame for fuzzy matching
    evaluator_df = pd.DataFrame(evaluator_annotations)
    all_annotations.extend(evaluator_annotations)
    
    # Second pass - process human data with fuzzy matching
    for task in tasks:
        # Process human data
        task_human = human_df[human_df['dim'] == task].copy()
        if len(task_human) > 0:
            task_human['score'] = pd.to_numeric(task_human['score'], errors='coerce')
            task_human = task_human.dropna(subset=['score'])
            task_human['prompt_clean'] = task_human['prompt'].apply(clean_prompt_for_matching)
            task_human['prompt_truncated'] = task_human['prompt_clean'].str[:100]
            
            # Generate initial prompt_id
            task_human['prompt_id'] = task_human.apply(
                lambda x: generate_prompt_id(x['prompt_clean'], x['subject_model']), axis=1
            )
            
            # Check if this prompt_id exists in evaluator data
            # If not, try fuzzy matching
            for idx, row in task_human.iterrows():
                prompt_id = row['prompt_id']
                model = row['subject_model']
                
                # Check if this prompt_id exists in evaluator data
                exists = ((evaluator_df['prompt_id'] == prompt_id) & 
                         (evaluator_df['subject_model'] == model)).any()
                
                if not exists:
                    # Try fuzzy matching
                    fuzzy_id = find_fuzzy_match_id(
                        row['prompt_clean'], 
                        model, 
                        evaluator_df[evaluator_df['task'] == task],
                        threshold=99
                    )
                    if fuzzy_id:
                        print(f"    Fuzzy matched human prompt for {model} in {task}")
                        task_human.at[idx, 'prompt_id'] = fuzzy_id
            
            # Add human annotations to unified dataset
            for _, row in task_human.iterrows():
                all_annotations.append({
                    'task': task,
                    'evaluator': 'human',
                    'annotator_id': str(row.get('prolific_id_hash', 'unknown')) if not pd.isna(row.get('prolific_id_hash')) else 'unknown',
                    'subject_model': row['subject_model'],
                    'prompt_id': row['prompt_id'],
                    'prompt_truncated': row['prompt_truncated'],
                    'prompt_clean': row['prompt_clean'],
                    'prompt_original': row['prompt'],
                    'subject_response': row.get('subject_response', ''),
                    'score': row['score']
                })
    
    # Save the unified dataset
    if all_annotations:
        unified_df = pd.DataFrame(all_annotations)
        unified_file = base_path / 'krippendorff_unified_annotations.csv'
        unified_df.to_csv(unified_file, index=False)
        print(f"\nUnified dataset saved to: {unified_file.name}")
        print(f"  Total annotations: {len(unified_df):,}")
        print(f"  Human annotations: {len(unified_df[unified_df['evaluator'] == 'human']):,}")
        print(f"  Gemini annotations: {len(unified_df[unified_df['evaluator'] == 'gemini']):,}")
        print(f"  Claude annotations: {len(unified_df[unified_df['evaluator'] == 'claude']):,}")
        print(f"  O3 annotations: {len(unified_df[unified_df['evaluator'] == 'o3']):,}")
        print(f"  4.1 annotations: {len(unified_df[unified_df['evaluator'] == '4.1']):,}")
        print(f"  Unique prompt IDs: {unified_df['prompt_id'].nunique():,}")
        print(f"  Tasks: {', '.join(unified_df['task'].unique())}")
    
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
                        
                        # Keep essential columns including new ID, truncated prompt, and human annotator ID
                        columns_to_keep = ['task', 'evaluator', 'subject_model', 'prompt_id', 
                                         'prompt_truncated', 'prompt', 'human_score', 'ai_score',
                                         'prolific_id_hash']  # Include human annotator ID
                        
                        # Only include columns that exist in matched
                        available_columns = [col for col in columns_to_keep if col in matched.columns]
                        matched = matched[available_columns]
                        all_matched_data.append(matched)
    
    if all_matched_data:
        matched_df = pd.concat(all_matched_data, ignore_index=True)
        matched_file = base_path / 'krippendorff_matched_individual_annotations.csv'
        matched_df.to_csv(matched_file, index=False)
        print(f"  Matched individual annotations saved to: {matched_file.name}")
        print(f"  Total matched annotations: {len(matched_df):,}")
        
        replication_data = []
        
        for task in tasks:
            task_matches = matched_df[matched_df['task'] == task]
            if len(task_matches) > 0:
                # Group by prompt_id and subject_model to get all annotations per prompt-model pair
                for (prompt_id, subject_model), group in task_matches.groupby(['prompt_id', 'subject_model']):
                    # Get all human annotators and their scores for this prompt-model pair
                    human_annotations = []
                    if 'prolific_id_hash' in group.columns:
                        for _, row in group.iterrows():
                            annotator_id = row.get('prolific_id_hash', 'unknown')
                            # Handle NaN or float values
                            if pd.isna(annotator_id):
                                annotator_id = 'unknown'
                            else:
                                annotator_id = str(annotator_id)
                            human_annotations.append({
                                'annotator_id': annotator_id,
                                'score': row['human_score']
                            })
                    
                    # Create a summary row with all information
                    summary_row = {
                        'prompt_id': prompt_id,
                        'subject_model': subject_model,
                        'task': task,
                        'prompt_truncated': group['prompt_truncated'].iloc[0],
                        'prompt_full': group['prompt'].iloc[0],
                        'num_human_annotations': len(human_annotations),
                        'human_scores': ','.join([str(a['score']) for a in human_annotations]),
                        'human_annotator_ids': ','.join([a['annotator_id'] for a in human_annotations])
                    }
                    
                    # Add AI evaluator scores
                    for evaluator in ['gemini', 'claude', 'o3', '4.1']:
                        eval_scores = group[group['evaluator'] == evaluator]['ai_score'].values
                        if len(eval_scores) > 0:
                            summary_row[f'{evaluator}_score'] = eval_scores[0]
                        else:
                            summary_row[f'{evaluator}_score'] = None
                    
                    replication_data.append(summary_row)
        
        if replication_data:
            replication_df = pd.DataFrame(replication_data)
            replication_file = base_path / 'krippendorff_replication_dataset.csv'
            replication_df.to_csv(replication_file, index=False)
            print(f"  Replication dataset saved to: {replication_file.name}")
            print(f"  Contains {len(replication_df):,} unique prompt-model-task combinations")
            print(f"  Key columns:")
            print(f"    - prompt_id: Unique identifier for prompt-model pair")
            print(f"    - human_annotator_ids: Comma-separated list of annotator IDs")  
            print(f"    - human_scores: Comma-separated list of human scores")
            print(f"    - [evaluator]_score: Score from each AI evaluator")
            print(f"  Use prompt_id to match experiments across runs")

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