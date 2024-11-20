import os.path as osp
import yaml
import os
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy

import pandas as pd
import numpy as np
import umap.umap_ as umap
from plotly import graph_objects as go
import argh

from src.utils import setup_keys, pass_optional_params, create_collapsible_html_list
from src.dataset_generation import generate_dataset
from src.evaluate_prompt_quality import calculate_prompt_scores
from src.diversity import evaluate_prompt_diversity
from src.evaluate_model import evaluate_model
from src.prompts import prompt_objects

SPEC_FILE = "spec.yaml"
KEYS_PATH = "keys.json"
PLOT_HEIGHTS = 600
N_CONCURRENTLY_EVALUATED_SUBJECTS = 50


def load_config(folder: str) -> dict:
    config_path = os.path.join(folder, SPEC_FILE)
    with open(config_path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def calculate_and_visualize_scores(prompts: list, system_prompts, generative_prompts, config: dict) -> tuple:

    relevance_score, relevance_system_prompt, relevance_prompt, passed_evaluation = calculate_prompt_scores(
        prompts,
        **pass_optional_params(general_params=config['general_params'], params=config['QA_params'])
    )

    df = pd.DataFrame({
        'prompt': prompts,
        'system_prompt': relevance_system_prompt,
        'generative_prompt': generative_prompts,
        'relevance_score': relevance_score,
        'relevance_prompt': relevance_prompt,
        'relevance_system_prompt': relevance_system_prompt
    })
    df = df.sort_values('relevance_score', ascending=False)
    plot_data = {
        "Generated_prompts": {
            x['prompt']: [
                'System Prompt: ' + x['system_prompt'],
                'Genrative Prompt: ' + x['generative_prompt'],
                {
                    f'Relevance: {x["relevance_score"]}': [
                        f'Relevance prompt: {x["relevance_prompt"]}'
                    ]
                }
            ]
            for _, x in df.iterrows()
        }
    }

    hist_fig = go.Figure(
        go.Histogram(x=relevance_score, name='Relevance')
    )
    hist_fig.update_layout(
        title="Prompt Evaluation Scores Histogram",
        xaxis_title="Score",
        yaxis_title="Frequency",
        height=PLOT_HEIGHTS
    )
    scatter_fig = go.Figure(
        go.Scatter(
            x=list(range(len(relevance_score))),
            y=relevance_score,
            mode='markers',
            text=prompts,
            hovertemplate='<b>Score</b>: %{y}<br>' +
                        '<b>Prompt</b>: %{text}<br>' +
                        '<extra></extra>',
        )
    )
    scatter_fig.update_layout(
        title="Prompt Evaluation Scores Scatter Plot",
        xaxis_title="Prompt Index",
        yaxis_title="Score (1-1000)",
        hovermode='closest',
        height=600,
        width=1200,
    )


    html_buffer = StringIO()
    hist_fig.write_html(html_buffer, full_html=False)
    scatter_fig.write_html(html_buffer, full_html=False)
    html_str = create_collapsible_html_list(plot_data) + html_buffer.getvalue()

    return relevance_score, relevance_system_prompt, relevance_prompt, html_str, passed_evaluation


def create_dataframe(prompts, system_prompts, generative_prompts, relevance_scores, passed_evaluation,
                     relevance_system_prompts, relevance_prompts) -> pd.DataFrame:

        return pd.DataFrame({
            'prompt': prompts,
            'system_prompt': system_prompts,
            'generative_prompt': generative_prompts,
            'relevance_score': relevance_scores,
            'passed_evaluation': passed_evaluation,
            'relevance_system_prompt': relevance_system_prompts,
            'relevance_prompt': relevance_prompts
        })


def evaluate_and_visualize_diversity(passed_qa_df: pd.DataFrame, config: dict) -> tuple:
    embeddings, pca_features, cluster, representative_samples, is_representative = evaluate_prompt_diversity(
        passed_qa_df['prompt'].tolist(),
        **pass_optional_params(general_params=config['general_params'], params=config['diversity_params'])
    )

    umap_reducer = umap.UMAP(n_components=2, random_state=42)
    vis_dims = umap_reducer.fit(np.array(pca_features)).embedding_

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=vis_dims[:, 0], y=vis_dims[:, 1], mode='markers',
            text=passed_qa_df['prompt'], hoverinfo='text',
            marker=dict(color=cluster, colorscale='Viridis', size=5, opacity=0.7),
            name='Non Representative Prompts'
        )
    )
    fig.add_trace(
        go.Scatter(
            x=vis_dims[representative_samples, 0], y=vis_dims[representative_samples, 1], mode='markers',
            text=passed_qa_df['prompt'][representative_samples], hoverinfo='text',
            marker=dict(color='red', size=10, opacity=0.9),
            name='Representative Prompts'
        )
    )
    fig.update_layout(
        title="UMAP Visualization of Text Embeddings with k-Means Clustering",
        xaxis_title="Component 1", yaxis_title="Component 2", hovermode='closest', height=PLOT_HEIGHTS
    )

    html_buffer = StringIO()
    fig.write_html(html_buffer, full_html=False)
    html_str = html_buffer.getvalue()

    return is_representative, html_str


def create_representative_prompts_html(is_diverse_df: pd.DataFrame) -> str:
    plot_data = {
        "Representative Prompts ": {
            x['prompt']: [
                'system_prompt: ' + x['system_prompt'],
                'generative_prompt: ' + x['generative_prompt'],
                {
                    f'relevance_score: {x["relevance_score"]}': [
                        f'relevance_system_prompt: {x["relevance_system_prompt"]}',
                        f'relevance_prompt: {x["relevance_prompt"]}'
                    ],
                },
            ]
            for _, x in is_diverse_df.iterrows()
        }
    }
    return create_collapsible_html_list(plot_data)


def create_subject_responses_html(is_diverse_df: pd.DataFrame, subject_model, best_possible_score) -> str:
    plot_data = {
        f"Subject responses ({subject_model})": [
            f"Best possible score: {best_possible_score}",
            [
                {
                    f"Score: {x['score']}": [
                        f"Prompt: {x['prompt']}",
                        # f"Subject response: {x['subject_response']}",
                        f"Subject system prompt: {x['subject_system_prompt']}",
                        f"Evaluator system prompt: {x['evaluator_system_prompt']}",
                        f"Evaluator prompt: {x['evaluator_prompt']}"
                        f"Evaluator response: {x['evaluator_response']}"
                    ]
                }
                for _, x in is_diverse_df.sort_values('score', ascending=False).iterrows()
            ]
        ]
    }

    return create_collapsible_html_list(plot_data)


def evaluate_and_visualize_model(is_diverse_df: pd.DataFrame, config: dict) -> str:

    html_out = ""
    html_scores = ""
    fig = go.Figure()

    subject_models = config['evaluation_params']['subject_models']
    del config['evaluation_params']['subject_models']

    dfs = []

    with ThreadPoolExecutor(max_workers=N_CONCURRENTLY_EVALUATED_SUBJECTS) as executor:

        future_to_index = {}
        for subject_model in subject_models:
            config = deepcopy(config)
            config['evaluation_params']['subject_model'] = subject_model

            future_to_index.update({
                executor.submit(
                    evaluate_model,
                    is_diverse_df['prompt'].tolist(),
                    **pass_optional_params(general_params=config['general_params'], params=config['evaluation_params'])
                ): subject_model
            })

        for future in as_completed(future_to_index):
            scores, subject_responses, subject_system_prompts, evaluator_system_prompts, evaluator_prompts, evaluator_responses = future.result()

            is_diverse_df = is_diverse_df.copy()

            is_diverse_df['score'] = scores
            is_diverse_df['subject_response'] = subject_responses
            is_diverse_df['subject_system_prompt'] = subject_system_prompts
            is_diverse_df['evaluator_system_prompt'] = evaluator_system_prompts
            is_diverse_df['evaluator_prompt'] = evaluator_prompts
            is_diverse_df['evaluator_response'] = evaluator_responses
            is_diverse_df['subject_model'] = future_to_index[future]


            dfs.append(is_diverse_df)

            best_possible_score = prompt_objects[config['general_params']['problem_type']]().get_top_eval_score()

            fig.add_trace(
                go.Histogram(x=scores, name=f"{future_to_index[future]}")
            )

            html_out += create_subject_responses_html(is_diverse_df, future_to_index[future], best_possible_score)
            html_scores += f"<h3>{future_to_index[future]} Mean Score: {np.mean(scores) / best_possible_score * 100:.2f}%</h3>"


    fig.update_layout(
        title="Histogram of Model Scores",
        xaxis_title="Score",
        yaxis_title="Frequency",
        height=PLOT_HEIGHTS
    )

    html_str = StringIO()
    fig.write_html(html_str, full_html=False)
    html_out += html_str.getvalue()

    df = pd.concat(dfs, ignore_index=True)

    return html_out, html_scores, df


def pipeline(folder: str):
    setup_keys(KEYS_PATH)
    config = load_config(folder)

    html_out = f"<h1>Eval generation phase</h1>"
    model_scores_html = ""
    
    if "general_params" in config:
        prompts, system_prompts, generative_prompts = generate_dataset(
            **pass_optional_params(general_params=config['general_params'], params=config['generation_params'])
        )

        if "QA_params" in config:
            scores_results = calculate_and_visualize_scores(prompts, system_prompts, generative_prompts, config)
            relevance_scores, relevance_system_prompts, relevance_prompts, scores_html, passed_evaluation = scores_results
            html_out += scores_html

            df = create_dataframe(prompts, system_prompts, generative_prompts, relevance_scores, passed_evaluation,
                                    relevance_system_prompts, relevance_prompts)

            passed_qa_df = df[df['passed_evaluation']].reset_index(drop=True)

            if "diversity_params" in config:

                is_representative, diversity_html = evaluate_and_visualize_diversity(passed_qa_df, config)
                html_out += diversity_html

                is_diverse_df = passed_qa_df[is_representative].reset_index(drop=True)

                html_out += create_representative_prompts_html(is_diverse_df)

                if "evaluation_params" in config:

                    model_evaluation_html, model_scores_html, out_df = evaluate_and_visualize_model(is_diverse_df, config)
                    html_out += "<h1>Model evaluation phase</h1>"
                    html_out += model_evaluation_html

                    with open(os.path.join(folder, 'raw.csv'), 'w') as f:
                        f.write(out_df.to_csv(index=False))
                    
                    prompts_df = out_df[['prompt']].copy()
                    with open(os.path.join(folder, 'samples.csv'), 'w') as f:
                        f.write(prompts_df.to_csv(index=False))


    html_out = f"<h1>{config['general_params']['problem_type']}</h1>" + \
        f"<h1>{os.path.split(folder)[-1]}</h1>" + model_scores_html + html_out

    with open(os.path.join(folder, 'plot.html'), 'w') as f:
        f.write(html_out)

if __name__ == '__main__':
    argh.dispatch_command(pipeline)