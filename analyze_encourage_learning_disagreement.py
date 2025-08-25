#!/usr/bin/env python3
"""
Analyze disagreement on issues for encourage_learning across different evaluators.
"""

import pandas as pd
from pathlib import Path
import json
from collections import defaultdict, Counter
import numpy as np

def load_evaluator_data(evaluator_path, task='encourage_learning'):
    """Load raw.csv for a specific evaluator and task."""
    csv_path = Path(evaluator_path) / task / 'raw.csv'
    if not csv_path.exists():
        print(f"Warning: {csv_path} not found")
        return None
    
    df = pd.read_csv(csv_path)
    return df

def parse_issues(evaluator_response):
    """Parse issues from evaluator response JSON."""
    try:
        # Try to parse as JSON
        response_data = json.loads(evaluator_response)
        if isinstance(response_data, dict) and 'issues' in response_data:
            return set(response_data['issues'])
        return set()
    except:
        # If not valid JSON, try to find issues pattern
        if isinstance(evaluator_response, str):
            # Look for patterns like "issues": ["A", "B", "C"]
            import re
            match = re.search(r'"issues"\s*:\s*\[(.*?)\]', evaluator_response)
            if match:
                issues_str = match.group(1)
                # Extract letters between quotes
                issues = re.findall(r'"([A-Z])"', issues_str)
                return set(issues)
        return set()

def analyze_disagreements():
    """Main analysis function."""
    
    # Define evaluator paths
    evaluators = {
        'gemini': 'output/openrouter_gemini_eval',
        'claude_4_sonnet': 'output/claude_4_sonnet_o3_continued',
        'o3': 'output/4_1_generated_o3_evaluated',
        '4.1': 'output/4_1_o3_continued'
    }
    
    # Load data from all evaluators
    all_data = {}
    for name, path in evaluators.items():
        df = load_evaluator_data(path)
        if df is not None:
            all_data[name] = df
            print(f"Loaded {len(df)} rows for {name}")
    
    if not all_data:
        print("No data loaded!")
        return
    
    # Find common prompts across all evaluators
    common_prompts = None
    for name, df in all_data.items():
        prompts = set(df['prompt'].values)
        if common_prompts is None:
            common_prompts = prompts
        else:
            common_prompts = common_prompts.intersection(prompts)
    
    print(f"\nFound {len(common_prompts)} common prompts across all evaluators")
    
    # Analyze disagreements
    disagreements = []
    issue_frequency = defaultdict(lambda: defaultdict(int))
    
    for prompt in common_prompts:
        prompt_data = {'prompt': prompt}
        issues_by_evaluator = {}
        
        # Get issues for each evaluator
        for name, df in all_data.items():
            row = df[df['prompt'] == prompt].iloc[0]
            issues = parse_issues(row['evaluator_response'])
            issues_by_evaluator[name] = issues
            prompt_data[f'{name}_issues'] = list(issues)
            prompt_data[f'{name}_score'] = row['score']
            
            # Track issue frequency
            for issue in issues:
                issue_frequency[issue][name] += 1
        
        # Calculate disagreement metrics
        all_issues = set()
        for issues in issues_by_evaluator.values():
            all_issues.update(issues)
        
        # Count how many evaluators agree on each issue
        issue_agreement = {}
        for issue in all_issues:
            count = sum(1 for issues in issues_by_evaluator.values() if issue in issues)
            issue_agreement[issue] = count
        
        # Measure disagreement
        prompt_data['total_unique_issues'] = len(all_issues)
        prompt_data['disagreement_score'] = len(all_issues) - max(issue_agreement.values()) if issue_agreement else 0
        prompt_data['issue_agreement'] = issue_agreement
        
        disagreements.append(prompt_data)
    
    # Sort by disagreement score
    disagreements.sort(key=lambda x: x['disagreement_score'], reverse=True)
    
    # Print top disagreements
    print("\n=== TOP 10 PROMPTS WITH HIGHEST DISAGREEMENT ===")
    for i, item in enumerate(disagreements[:10]):
        print(f"\n{i+1}. Prompt: {item['prompt'][:100]}...")
        print(f"   Disagreement Score: {item['disagreement_score']}")
        print("   Issues by evaluator:")
        for name in evaluators.keys():
            issues = item.get(f'{name}_issues', [])
            score = item.get(f'{name}_score', 'N/A')
            print(f"   - {name}: {issues} (score: {score})")
        print(f"   Issue agreement: {item['issue_agreement']}")
    
    # Print issue frequency analysis
    print("\n=== ISSUE FREQUENCY BY EVALUATOR ===")
    all_issues = sorted(issue_frequency.keys())
    
    # Create a table
    print(f"{'Issue':<10}", end='')
    for name in evaluators.keys():
        print(f"{name:<20}", end='')
    print()
    print("-" * (10 + 20 * len(evaluators)))
    
    for issue in all_issues:
        print(f"{issue:<10}", end='')
        for name in evaluators.keys():
            count = issue_frequency[issue][name]
            pct = (count / len(common_prompts)) * 100
            print(f"{count} ({pct:5.1f}%)        ", end='')
        print()
    
    # Calculate pairwise agreement
    print("\n=== PAIRWISE AGREEMENT ON ISSUES ===")
    evaluator_names = list(evaluators.keys())
    
    for i in range(len(evaluator_names)):
        for j in range(i+1, len(evaluator_names)):
            eval1, eval2 = evaluator_names[i], evaluator_names[j]
            agreements = 0
            total = 0
            
            for prompt_data in disagreements:
                issues1 = set(prompt_data.get(f'{eval1}_issues', []))
                issues2 = set(prompt_data.get(f'{eval2}_issues', []))
                
                # Count exact matches
                if issues1 == issues2:
                    agreements += 1
                total += 1
            
            agreement_pct = (agreements / total) * 100 if total > 0 else 0
            print(f"{eval1} vs {eval2}: {agreement_pct:.1f}% exact agreement ({agreements}/{total})")
    
    # Save detailed results
    results_df = pd.DataFrame(disagreements)
    results_df.to_csv('encourage_learning_disagreement_analysis.csv', index=False)
    print(f"\nDetailed results saved to encourage_learning_disagreement_analysis.csv")
    
    return disagreements, issue_frequency

if __name__ == "__main__":
    analyze_disagreements()