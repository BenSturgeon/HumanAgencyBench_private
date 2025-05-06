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


    
    
    return results, plots


def generate_html_report(results, plots):
    """Generate HTML report with results and plots."""
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
    </body>
    </html>
    """

    return html

def main():
    
    config = load_config('sensitivity_analysis/inter_evaluator_agreement/inter_eval_sensitivity_config.yaml')

    evaluators = [
        # 'grok-3-beta',
        'models/gemini-2.5-pro-preview-03-25',
        # 'models/gemini-2.5-flash-preview-04-17',
        # 'claude-3-opus-20240229',
        'models/gemini-2.0-flash',
        'claude-3-7-sonnet-20250219',
        'claude-3-5-sonnet-20241022',
        'claude-3-5-sonnet-20240620',
        'gpt-4.1',
        'o4-mini', # Assuming 'o4-mini' is the correct alias for gpt-4o-mini
        # Removed duplicate 'gpt-4.1'
        'gpt-4o',
    ]
    alias = [
        # 'grok3b',
        'gemini25pro',
        # 'gemini25flash',
        # 'claude_opus',
        'gemini_2_flash',
        'claude37s',
        'claude35s_241022', # Shortened alias
        'claude35s_240620', # Shortened alias
        'gpt41',
        'o4mini',
        'gpt4o',
    ]
    for evaluator, alias_name in zip(evaluators, alias):

        conf_tmp = deepcopy(config)
        conf_tmp['evaluation_params']['evaluator_model'] = evaluator
        # Add an explicit max_tokens setting for the evaluator
        conf_tmp['evaluation_params']['evaluator_max_tokens'] = 2000 

        with open("/tmp/inter_eval_sensitivity_config.yaml", 'w') as f:
            f.write(yaml.dump(conf_tmp))

        conf_tmp_path = "/tmp/inter_eval_sensitivity_config.yaml"

        pipeline(conf_tmp_path, f"sensitivity_analysis/inter_evaluator_agreement/{alias_name}")

    # Update the dirs list to use the new aliases
    base_path = 'sensitivity_analysis/inter_evaluator_agreement'
    # Assuming 'acknowledge_limitations' is the task sub-directory.
    # If it's different, please let me know.
    task_subdir = 'acknowledge_limitations'
    dirs = [f"{base_path}/{a}/{task_subdir}" for a in alias]


    try:
        results, plots = analyze_inter_evaluator_agreement(dirs)
        report = generate_html_report(results, plots)

        with open('sensitivity_analysis/inter_evaluator_agreement/report_updated_models.html', 'w') as f:
            f.write(report)
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
