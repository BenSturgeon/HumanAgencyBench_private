import os.path as osp
import yaml
import os
from io import StringIO
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
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


def load_config(folder: str) -> dict:
    config_path = os.path.join(folder, SPEC_FILE)
    with open(config_path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def generate_and_visualize_dataset(config: dict) -> tuple:
    prompts, system_prompts, generative_prompts = generate_dataset(
        **pass_optional_params(general_params=config['general_params'], params=config['generation_params'])
    )
    
    plot_data = {
        "Generated Prompts ": { 
            prompt: [
                'System Prompt: ' + system_prompt,
                'Generative Prompt: ' + generative_prompt
            ]
            for prompt, system_prompt, generative_prompt in zip(prompts, system_prompts, generative_prompts)
        }
    }
    
    html_out = create_collapsible_html_list(plot_data)
    return prompts, system_prompts, generative_prompts, html_out


def calculate_and_visualize_scores(prompts: list, config: dict) -> tuple:
    correctness_scores, relevance_scores, harmonic_mean_scores, passed_evaluation, relevance_system_prompts, \
        correctness_system_prompts, relevance_prompts, correctness_prompts = calculate_prompt_scores(
            prompts, 
            **pass_optional_params(general_params=config['general_params'], params=config['QA_params'])
    )

    fig = go.Figure(
        [
            go.Histogram(x=correctness_scores, name='Correctness'),
            go.Histogram(x=relevance_scores, name='Relevance'),
            go.Histogram(x=harmonic_mean_scores, name='Harmonic Mean')
        ]
    )
    fig.update_layout(
        title="Histogram of Prompt Scores",
        xaxis_title="Score",
        yaxis_title="Frequency",
        height=PLOT_HEIGHTS
    )

    html_buffer = StringIO()
    fig.write_html(html_buffer, full_html=False)
    html_str = html_buffer.getvalue()
    
    return correctness_scores, relevance_scores, harmonic_mean_scores, passed_evaluation, \
           relevance_system_prompts, correctness_system_prompts, relevance_prompts, correctness_prompts, html_str


def create_dataframe(prompts, system_prompts, generative_prompts, correctness_scores, relevance_scores, 
                     harmonic_mean_scores, passed_evaluation, relevance_system_prompts, correctness_system_prompts, 
                     relevance_prompts, correctness_prompts) -> pd.DataFrame:
    return pd.DataFrame({
        'prompt': prompts,
        'system_prompt': system_prompts,
        'generative_prompt': generative_prompts,
        'correctness_score': correctness_scores,
        'relevance_score': relevance_scores,
        'harmonic_mean_score': harmonic_mean_scores,
        'passed_evaluation': passed_evaluation,
        'relevance_system_prompt': relevance_system_prompts,
        'correctness_system_prompt': correctness_system_prompts,
        'relevance_prompt': relevance_prompts,
        'correctness_prompt': correctness_prompts
    })


def evaluate_and_visualize_diversity(passed_qa_df: pd.DataFrame, config: dict) -> tuple:
    embeddings, pca_features, cluster, representative_samples, is_representative = evaluate_prompt_diversity(
        passed_qa_df['prompt'].tolist(),
        **pass_optional_params(general_params=config['general_params'], params=config['diversity_params'])
    )

    tsne = TSNE(n_components=2, random_state=42, init='random', learning_rate=200)
    vis_dims = tsne.fit_transform(np.array(pca_features))

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
        title="t-SNE Visualization of Text Embeddings with k-Means Clustering",
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
                    f'correctness_score: {x["correctness_score"]}': [
                        f'correctness_system_prompt: {x["correctness_system_prompt"]}',
                        f'correctness_prompt: {x["correctness_prompt"]}'
                    ],
                    f'relevance_score: {x["relevance_score"]}': [
                        f'relevance_system_prompt: {x["relevance_system_prompt"]}',
                        f'relevance_prompt: {x["relevance_prompt"]}'
                    ],
                },
                f'harmonic_mean_score: {x["harmonic_mean_score"]}'
            ]
            for _, x in is_diverse_df.iterrows()
        }
    }
    return create_collapsible_html_list(plot_data)

def create_subject_responses_html(is_diverse_df: pd.DataFrame, subject_model) -> str:
    plot_data = {
        f"Subject responses ({subject_model})": {
            f"Prompt:\n\n\n{x['prompt']}\n\n\nSubject response:\n\n\n{x['subject_responses']}": [
                'subject_system_prompt: ' + x['subject_system_prompts'],
                'subject_prompt: ' + x['prompt'],
                'evaluator_system_prompt: ' + x['evaluator_system_prompts'],
                'evaluator_prompt: ' + x['evaluator_prompts'],
                f'score: {x["score"]}'
            ]
            for _, x in is_diverse_df.sort_values('score', ascending=False).iterrows()
        }
    }
    return create_collapsible_html_list(plot_data)


def evaluate_and_visualize_model(is_diverse_df: pd.DataFrame, config: dict) -> str:

    html_out = ""
    html_append = ""
    fig = go.Figure()

    subject_models = config['evaluation_params']['subject_models']
    del config['evaluation_params']['subject_models']

    for subject_model in subject_models:
        config['evaluation_params']['subject_model'] = subject_model
        
        scores, subject_responses, subject_system_prompts, evaluator_system_prompts, evaluator_prompts = evaluate_model(
            is_diverse_df['prompt'].tolist(),
            **pass_optional_params(general_params=config['general_params'], params=config['evaluation_params'])
        )

        is_diverse_df['score'] = scores
        is_diverse_df['subject_responses'] = subject_responses
        is_diverse_df['subject_system_prompts'] = subject_system_prompts
        is_diverse_df['evaluator_system_prompts'] = evaluator_system_prompts
        is_diverse_df['evaluator_prompts'] = evaluator_prompts
        

        html_out += create_subject_responses_html(is_diverse_df, config['evaluation_params']['subject_model'])

        fig.add_trace(
            go.Histogram(x=scores, name=subject_model)
        )

        best_possible_score = prompt_objects[config['general_params']['problem_type']]().get_top_eval_score()
        html_append += f"<h3>{subject_model} Mean Score: {np.mean(scores) / best_possible_score * 100:.2f}%</h3>"

    fig.update_layout(
        title="Histogram of Model Scores",
        xaxis_title="Score",
        yaxis_title="Frequency",
        height=PLOT_HEIGHTS
    )

    html_str = StringIO()
    fig.write_html(html_str, full_html=False)
    html_out += html_str.getvalue()

    html_out += html_append

    return html_out


def pipeline(folder: str):
    setup_keys(KEYS_PATH)
    config = load_config(folder)
    
    html_out = f"<h1>{os.path.split(folder)[-1]}</h1>"

    if "general_params" in config:
        prompts, system_prompts, generative_prompts, dataset_html = generate_and_visualize_dataset(config)
        html_out += dataset_html


        if "QA_params" in config:
            scores_results = calculate_and_visualize_scores(prompts, config)
            correctness_scores, relevance_scores, harmonic_mean_scores, passed_evaluation, \
            relevance_system_prompts, correctness_system_prompts, relevance_prompts, correctness_prompts, scores_html = scores_results
            html_out += scores_html

            df = create_dataframe(prompts, system_prompts, generative_prompts, correctness_scores, relevance_scores, 
                                harmonic_mean_scores, passed_evaluation, relevance_system_prompts, correctness_system_prompts, 
                                relevance_prompts, correctness_prompts)

            passed_qa_df = df[df['passed_evaluation']].reset_index(drop=True)

            if "diversity_params" in config:

                is_representative, diversity_html = evaluate_and_visualize_diversity(passed_qa_df, config)
                html_out += diversity_html

                is_diverse_df = passed_qa_df[is_representative].reset_index(drop=True)

                html_out += create_representative_prompts_html(is_diverse_df)

                if "evaluation_params" in config:

                    model_evaluation_html = evaluate_and_visualize_model(is_diverse_df, config)
                    html_out += model_evaluation_html

    with open(os.path.join(folder, 'plot.html'), 'w') as f:
        f.write(html_out)

if __name__ == '__main__':
    argh.dispatch_command(pipeline)
