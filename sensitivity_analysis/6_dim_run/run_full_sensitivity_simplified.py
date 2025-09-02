#!/usr/bin/env python3
"""Run full rubric sensitivity analysis with 250 samples for all evaluators and all rubrics."""

import pandas as pd
import json
import time
from pathlib import Path
import sys
import os
import concurrent.futures

print("Starting sensitivity analysis script...", flush=True)

# Add paths
sys.path.insert(0, '/home/team/agency_evaluations')
sys.path.insert(0, '/home/team/agency_evaluations/sensitivity_analysis')

# Import after path is set
from src.utils import setup_keys
setup_keys("/home/team/agency_evaluations/keys.json")
print("Keys loaded", flush=True)

from sensitivity_analysis.rubric_sensitivity import (
    evaluate_single_pair_with_permutation,
    generate_permutation_unified,
    AvoidValueManipulation,
    RUBRIC_REGISTRY
)
print("Imports complete", flush=True)

# Configuration
SUBJECT_MODEL = 'three_models_combined'  # GPT-4o, Claude 3.7, Gemini 2.5 Flash
EVALUATORS = {
    'o3': 'openai/o3'  # Openrouter
}
NUM_RUNS_PER_SET = 3  # Number of variations per factor
FIXED_SEED = 0  # Seed for elements not being varied
MAX_WORKERS = 20  # 20 workers as requested

# Rubric mappings
RUBRIC_ABBREVIATIONS = {
    'acq': 'AskClarifyingQuestions',
    'avm': 'AvoidValueManipulation', 
    'cm': 'CorrectMisinformation',
    'did': 'DeferImportantDecisions',
    'el': 'EncourageLearning',
    'msb': 'MaintainSocialBoundaries'
}

# Task name mappings for file names
TASK_FILE_NAMES = {
    'acq': 'ask_clarifying_questions',
    'avm': 'avoid_value_manipulation',
    'cm': 'correct_misinformation',
    'did': 'defer_important_decisions',
    'el': 'encourage_learning',
    'msb': 'maintain_social_boundaries'
}

# Define variation sets
variation_sets = {
    "preamble": {"vary": "preamble_index", "fixed": ["issue_seed", "example_seed"]},
    "issues": {"vary": "issue_seed", "fixed": ["preamble_index", "example_seed"]},
    "examples": {"vary": "example_seed", "fixed": ["preamble_index", "issue_seed"]},
}

# Run for each rubric and evaluator
for rubric_abbr, rubric_name in RUBRIC_ABBREVIATIONS.items():
    print(f"\n{'='*80}", flush=True)
    print(f"Processing rubric: {rubric_name} ({rubric_abbr})", flush=True)
    print('='*80, flush=True)
    
    # Load combined CSV with 1500 samples (500 each from GPT-4o, Claude 3.7, Gemini 2.5 Flash)
    task_file_name = TASK_FILE_NAMES[rubric_abbr]
    baseline_path = Path(f"/home/team/agency_evaluations/sensitivity_analysis/three_models_o3_eval/{task_file_name}_combined_1500.csv")
    if not baseline_path.exists():
        print(f"Baseline file not found: {baseline_path}, skipping...")
        continue
    
    baseline_df = pd.read_csv(baseline_path)
    
    # Rename columns to match expected format
    if 'subject_response' in baseline_df.columns:
        baseline_df = baseline_df.rename(columns={'subject_response': 'response'})
    
    print(f"Loaded {len(baseline_df)} baseline responses for {rubric_abbr}", flush=True)
    print(f"  - Models: {baseline_df['model_group'].value_counts().to_dict() if 'model_group' in baseline_df.columns else 'N/A'}", flush=True)
    
    # Get rubric configuration from registry
    if rubric_name not in RUBRIC_REGISTRY:
        print(f"Rubric {rubric_name} not in registry, skipping...")
        continue
    
    rubric_def = RUBRIC_REGISTRY[rubric_name]
    
    # Create prompt object with registry data
    prompt_object = AvoidValueManipulation()  # Use as template
    prompt_object.issues = rubric_def.issues_base
    prompt_object.deductions = rubric_def.deductions_map_base_text_to_value
    prompt_object.top_eval_score = rubric_def.top_eval_score
    
    for eval_name, eval_model in EVALUATORS.items():
        print(f"\n--- Evaluator: {eval_name} ---", flush=True)
        print(f"Model: {eval_model}", flush=True)
        
        # Use MAX_WORKERS constant
        max_workers = MAX_WORKERS
        
        overall_start = time.time()
        output_base_dir = Path(f"results_three_models_o3/{rubric_abbr}_{eval_name}")
        
        # Run each variation set
        for set_name, config in variation_sets.items():
            print(f"\n  Varying {set_name}:", flush=True)
            
            for run_index in range(NUM_RUNS_PER_SET):
                run_start_time = time.time()
                print(f"    Run {run_index}...", flush=True)
                
                # Determine parameters for this run
                params = {}
                varying_param_name = config["vary"]
                params[varying_param_name] = run_index
                for fixed_param_name in config["fixed"]:
                    params[fixed_param_name] = FIXED_SEED
                
                # Generate the specific permutation
                selected_preamble, ih, examples_dict = generate_permutation_unified(rubric_name, **params)
                shuffled_formatted_issues = ih.get_shuffled_formated_issues()
                
                # Define output directory
                output_dir = output_base_dir / f"set_{set_name}" / f"run_{run_index}"
                output_dir.mkdir(parents=True, exist_ok=True)
                output_csv_path = output_dir / "evaluation_results.csv"
                
                # Check if already exists
                if output_csv_path.exists():
                    print(f"      Already exists, skipping: {output_csv_path}", flush=True)
                    continue
                
                print(f"      Output: {output_csv_path}", flush=True)
                
                # Evaluate all baseline pairs using this permutation
                all_results = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = []
                    
                    for idx, row in baseline_df.iterrows():
                        future = executor.submit(
                            evaluate_single_pair_with_permutation,
                            prompt_text=row['prompt'],
                            response_text=row['response'],
                            evaluator_model=eval_model,
                            preamble=selected_preamble,
                            shuffled_formatted_issues=shuffled_formatted_issues,
                            prompt_object=prompt_object,
                            temperature=0,
                            ih=ih,
                            examples_dict=examples_dict,
                            use_cache=False,  # Disabled caching for fresh evaluations
                            refresh_cache=False
                        )
                        futures.append((idx, future))
                    
                    # Collect results
                    for idx, future in futures:
                        try:
                            result = future.result(timeout=120)
                            all_results.append(result)
                            if (idx + 1) % 50 == 0:
                                print(f"        Processed {idx+1}/{len(baseline_df)}...", flush=True)
                        except Exception as e:
                            print(f"        Error on pair {idx}: {e}", flush=True)
                            all_results.append({
                                'prompt': baseline_df.iloc[idx]['prompt'],
                                'response': baseline_df.iloc[idx]['response'],
                                'score': -1,
                                'error': str(e)
                            })
                
                # Save results as CSV
                results_df = pd.DataFrame(all_results)
                results_df.to_csv(output_csv_path, index=False)
                
                run_time = time.time() - run_start_time
                valid_scores = results_df[results_df['score'] >= 0]['score']
                print(f"      Completed in {run_time:.1f}s ({len(valid_scores)} valid)", flush=True)
        
        overall_time = time.time() - overall_start
        print(f"  ✅ {eval_name} completed in {overall_time/60:.1f} minutes", flush=True)

print(f"\n{'='*80}", flush=True)
print("All sensitivity analyses completed!", flush=True)
print('='*80, flush=True)