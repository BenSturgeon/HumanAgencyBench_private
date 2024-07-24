import yaml
import os
from typing import List, Dict

import pandas as pd
import argh

from src.utils import setup_keys, pass_optional_params
from src.dataset_generation import generate_dataset
from src.evaluate_prompt_quality import calculate_prompt_scores
from src.diversity import evaluate_prompt_diversity
from src.evaluate_model import evaluate_model

SPEC_FILE = "spec.yaml"
KEYS_PATH = "keys.json"

def pipeline(folder: str):
    config_path = os.path.join(folder, SPEC_FILE)
    
    setup_keys(KEYS_PATH)

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    prompts: List[str] = generate_dataset(
        folder=folder, 
        **pass_optional_params(general_params=config['general_params'], params=config['generation_params'])
    )

    correctness_scores, relevance_scores, harmonic_mean_scores, passed_evaluation = calculate_prompt_scores(
        prompts, 
        **pass_optional_params(general_params=config['general_params'], params=config['QA_params'])
    )

    df = pd.DataFrame({
        'prompt': prompts,
        'correctness_score': correctness_scores,
        'relevance_score': relevance_scores,
        'harmonic_mean_score': harmonic_mean_scores,
        'passed_evaluation': passed_evaluation
    })

    passed_qa_df = df[df['passed_evaluation']].reset_index(drop=True)

    embeddings, pca_features, cluster, representative_samples, is_representative = evaluate_prompt_diversity(
        passed_qa_df['prompt'].tolist(),
        **pass_optional_params(general_params=config['general_params'], params=config['diversity_params'])
    )

    is_diverse_df = passed_qa_df[is_representative].reset_index(drop=True)

    print(is_diverse_df)

    scores, subject_responces = evaluate_model(
        is_diverse_df['prompt'].tolist(),
        **pass_optional_params(general_params=config['general_params'], params=config['evaluation_params'])
    )

    print("mean score for subject_model: ", config['evaluation_params']['subject_model'], "is: ", sum(scores) / len(scores))

if __name__ == '__main__':
    pipeline('cases/dummy_case')