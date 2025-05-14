import argparse
import pandas as pd
import yaml
from pathlib import Path
import plotly.graph_objects as go
from scipy import stats
from sklearn.metrics import cohen_kappa_score
from itertools import combinations
from datetime import datetime
from copy import deepcopy
import sys
import os 
import shutil

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Add the script's own directory explicitly to handle potential import issues
sensitivity_analysis_path = os.path.dirname(os.path.abspath(__file__))
if sensitivity_analysis_path not in sys.path:
    sys.path.insert(0, sensitivity_analysis_path) # Insert after root

from src.utils import load_config
from pipeline import pipeline


def load_and_validate_data(directory):
    """Load data from a directory and perform basic validation."""
    try:
        df = pd.read_csv(Path(directory) / "raw.csv")
        required_cols = {'prompt', 'subject_response', 'score', 'subject_model'}
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns in {directory}")
        return df
    except Exception as e:
        raise ValueError(f"Error loading data from {directory}: {str(e)}")


def get_evaluator_name(directory):
    """Extract evaluator name from directory path."""
    return Path(directory).parts[-2] 


def create_distribution_plot(scores_dict):
    """Create distribution plot of scores from different raters."""
    fig = go.Figure()
    for rater, scores in scores_dict.items():
        fig.add_trace(go.Histogram(x=scores, name=rater, opacity=0.7))
    fig.update_layout(
        title="Score Distributions by Rater",
        xaxis_title="Score",
        yaxis_title="Frequency",
        barmode='overlay'
    )
    return fig



def analyze_inter_evaluator_agreement(dirs):
    """Main analysis function."""
    # Load all datasets and get evaluator names
    dfs = []
    evaluator_names = []
    for i, dir_path in enumerate(dirs):
        df = load_and_validate_data(dir_path)
        df = df[['prompt', 'subject_response', 'score']]
        evaluator_name = get_evaluator_name(dir_path)
        evaluator_names.append(evaluator_name)
        if i > 0:
            df = df.rename(columns={'score': f'score_{i}'})
        dfs.append(df)
    
    # Merge all datasets
    merged_df = dfs[0]
    for i, df in enumerate(dfs[1:], 1):
        merged_df = pd.merge(
            merged_df,
            df,
            on=['prompt', 'subject_response']
        )

    print(merged_df.head())
    
    # Store scores with evaluator names
    score_cols = ['score'] + [f'score_{i}' for i in range(1, len(dfs))]
    scores = {
        name: merged_df[col] # Use the full evaluator name (alias) as the key
        for name, col in zip(evaluator_names, score_cols)
    }

    print("Scores dictionary keys:", scores.keys()) # Add print to confirm keys
    
    # Calculate pairwise differences and metrics
    results = {
        'pairs': [],
        'metrics': []
    }
    
    # Calculate differences and metrics for all pairs
    for (name1, score1), (name2, score2) in combinations(scores.items(), 2):

        print(name1, score1, name2, score2)
        diff = abs(score1 - score2)
        
        metrics = {
            'Pair': f'{name1} vs {name2}',
            'Exact Agreement': (diff == 0).mean(),
            'Within 1 Point': (diff <= 1).mean(),
            "Correlation": stats.pearsonr(score1, score2)[0],
            'Mean Difference': diff.mean(),
            'Std Difference': diff.std()
        }
        results['pairs'].append(f'{name1} vs {name2}')
        results['metrics'].append(metrics)
    
    # Create plots
    plots = {
        'distributions': create_distribution_plot(scores),
        'differences': {}
    }

    plots['differences'] = go.Figure()
    plots['differences'].update_layout(
        title="Score Differences",
        xaxis_title="Absolute Difference in Scores",
        yaxis_title="Frequency"
    )
    
    # Create difference plots for all pairs
    for (name1, score1), (name2, score2) in combinations(scores.items(), 2):
        diff = abs(score1 - score2)
        pair_name = f'{name1} vs {name2}'
        plots['differences'].add_trace(go.Histogram(x=diff, nbinsx=20, name=pair_name))

        tmp_fig = go.Figure()
        tmp_fig.add_trace(go.Histogram(x=diff, nbinsx=20, name=pair_name))
        # save
        tmp_fig.write_image(f"sensitivity_analysis/inter_evaluator_agreement/all.png")


    
    
    return results, plots, merged_df


def generate_html_report(results, plots, merged_df):
    """Generate HTML report with results and plots."""

    # Section for Individual Responses
    responses_html = '<div class="section">\n<h2>Individual Subject Responses</h2>\n'
    if merged_df is not None and not merged_df.empty:
        # Ensure 'prompt' and 'subject_response' columns exist
        if 'prompt' in merged_df.columns and 'subject_response' in merged_df.columns:
            for index, row in merged_df.iterrows():
                prompt = row['prompt']
                subject_response = row['subject_response']
                # Basic HTML escaping for display, especially for preformatted text
                prompt_display = prompt.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                response_display = subject_response.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

                responses_html += f'''
                <details>
                    <summary><b>Prompt:</b> {prompt_display[:150]}{'...' if len(prompt_display) > 150 else ''}</summary>
                    <div style="padding: 10px; border: 1px solid #eee; margin-top: 5px; background-color: #f9f9f9;">
                        <b>Full Prompt:</b>
                        <pre style="white-space: pre-wrap; word-wrap: break-word;">{prompt_display}</pre>
                        <b>Subject Response:</b>
                        <pre style="white-space: pre-wrap; word-wrap: break-word;">{response_display}</pre>
                    </div>
                </details>
                '''
        else:
            responses_html += "<p>Required columns ('prompt', 'subject_response') not found in the data.</p>"
    else:
        responses_html += "<p>No data available for individual responses.</p>"
    responses_html += '</div>'

    html = f"""
    <html>
    <head>
        <title>Sensitivity Analysis Report</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .section {{ margin-bottom: 30px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>Sensitivity Analysis Report</h1>
        <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="section">
            <h2>Score Distributions</h2>
            {plots['distributions'].to_html(full_html=False, include_plotlyjs=True)}
        </div>
        
        <div class="section">
            <h2>Agreement Metrics</h2>
            {pd.DataFrame(results['metrics']).to_html()}
        </div>
        
        <div class="section">
            <h2>Score Differences</h2>
            {plots['differences'].to_html(full_html=False, include_plotlyjs=False)}
        </div>

        {responses_html} 

    </body>
    </html>
    """

    return html

def main():
    
    config = load_config('sensitivity_analysis/inter_evaluator_agreement/inter_eval_sensitivity_config.yaml')
    evaluators = [
        'claude-3-5-sonnet-20240620',
        'models/gemini-2.5-pro-preview-03-25',
        'gpt-4.1',
        'grok-3-beta',
        'llama-4-maverick-instruct'
    ]
    alias = [
        'claude_3_5_sonnet',
        'gemini_2_5_pro',
        'gpt_4_1', 
        'grok_3',
        'llama_4_maverick'
    ]
    
    base_path = Path('sensitivity_analysis/inter_evaluator_agreement') # Use Path object

    for evaluator, alias_name in zip(evaluators, alias):
        conf_tmp = deepcopy(config)
        conf_tmp['evaluation_params']['evaluator_model'] = evaluator
        conf_tmp['evaluation_params']['evaluator_max_tokens'] = 2000 

        # Ensure the temp config directory exists (though /tmp usually does)
        temp_config_path = Path("/tmp/inter_eval_sensitivity_config.yaml")
        temp_config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_config_path, 'w') as f:
            f.write(yaml.dump(conf_tmp))

        output_dir_for_evaluator = base_path / alias_name
        # output_dir_for_evaluator.mkdir(parents=True, exist_ok=True) # pipeline should create this

        print(f"Running pipeline for {alias_name}...")
        pipeline(str(temp_config_path), str(output_dir_for_evaluator))

    print("\nAll pipeline runs complete. Discovering common task subdirectories...")

    # Discover common task subdirectories
    common_task_subdirs = None
    for i, evaluator_alias in enumerate(alias):
        current_evaluator_path = base_path / evaluator_alias
        if not current_evaluator_path.is_dir():
            print(f"Warning: Output directory for {evaluator_alias} not found: {current_evaluator_path}")
            continue

        evaluator_subdirs = {d.name for d in current_evaluator_path.iterdir() if d.is_dir()}
        
        if common_task_subdirs is None:
            common_task_subdirs = evaluator_subdirs
        else:
            common_task_subdirs.intersection_update(evaluator_subdirs)

    if not common_task_subdirs:
        print("Error: No common task subdirectories found across all evaluators. Exiting.")
        sys.exit(1)
    
    print(f"Found common task subdirectories: {common_task_subdirs}")

    for task_name in common_task_subdirs:
        print(f"\nProcessing task: {task_name}...")
        
        current_task_dirs = []
        valid_task = True
        for evaluator_alias in alias:
            task_dir_path = base_path / evaluator_alias / task_name
            raw_csv_path = task_dir_path / "raw.csv"
            if not raw_csv_path.exists():
                print(f"Warning: raw.csv not found in {raw_csv_path} for task {task_name}. Skipping this task for {evaluator_alias} or skipping task.")
                # Option: decide if task is entirely skipped or if analysis runs with fewer evaluators
                # For now, let's ensure all evaluators have data for this task.
                valid_task = False
                break 
            current_task_dirs.append(str(task_dir_path))
        
        if not valid_task:
            print(f"Skipping analysis for task '{task_name}' due to missing data for one or more evaluators.")
            continue

        try:
            print(f"Analyzing inter-evaluator agreement for task: {task_name} with dirs: {current_task_dirs}")
            results, plots, merged_df = analyze_inter_evaluator_agreement(current_task_dirs)
            
            # Generate and save HTML report
            report_filename = base_path / f'report_{task_name}.html'
            report = generate_html_report(results, plots, merged_df)
            with open(report_filename, 'w') as f:
                f.write(report)
            print(f"Saved HTML report to {report_filename}")

            # Save merged_df to CSV
            merged_csv_filename = base_path / f'merged_scores_{task_name}.csv'
            merged_df.to_csv(merged_csv_filename, index=False)
            print(f"Saved merged scores CSV to {merged_csv_filename}")

            # Copy individual raw.csv files
            print(f"Copying individual raw.csv files for task {task_name}...")
            for evaluator_alias in alias:
                source_raw_csv = base_path / evaluator_alias / task_name / "raw.csv"
                target_raw_csv = base_path / f'{evaluator_alias}_{task_name}_raw.csv'
                if source_raw_csv.exists():
                    shutil.copy(source_raw_csv, target_raw_csv)
                    print(f"Copied {source_raw_csv} to {target_raw_csv}")
                else:
                    print(f"Warning: Source raw.csv not found, cannot copy: {source_raw_csv}")

        except Exception as e:
            print(f"Error during analysis or reporting for task {task_name}: {str(e)}")
            # Continue to the next task if one fails
            continue
    
    print("\nProcessing complete.")

if __name__ == "__main__":
    main()
