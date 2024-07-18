import json
import os
from typing import List, Dict, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from src.llms import LLM
import numpy as np
from src.prompts import prompt_functions
from src.llms import OpenAILLM, AnthropicLLM, LocalLlama

from plotly import graph_objects as go
import json
import random
import pandas as pd

 
# PROMPTS_DATASET = 'datasets/needs_more_info/not_enough_info_to_answer/generated_prompts_raw_openai4o_t14.json'
# OUTPUT_PATH = 'datasets/needs_more_info/not_enough_info_to_answer/generated_prompts_with_relevance_scores.csv'
PROMPTS_DATASET = 'datasets/needs_more_info/not_enough_info_to_answer/not_enough_info_to_answer_human_expert.json'
OUTPUT_PATH = 'datasets/needs_more_info/not_enough_info_to_answer/not_enough_info_to_answer_human_expert_with_relevance_scores.csv'

LLM_API: Union['AnthropicLLM', 'OpenAILLM', 'LocalLlama'] = 'OpenAI'
MODEL = 'gpt-4o'
KEYS_PATH = "keys.json"
N_CONCURRENT_REQUESTS = 500

LLM = {
    'AnthropicLLM': AnthropicLLM,
    'OpenAI': OpenAILLM,
    'LocalLlama': LocalLlama
}[LLM_API]

keys = json.load(open(KEYS_PATH))
assert keys['OPENAI_API_KEY'] is not None if LLM_API == 'OpenAI' else True
assert keys['ANTHROPIC_API_KEY'] is not None if LLM_API == 'AnthropicLLM' else True
os.environ["OPENAI_API_KEY"] = keys['OPENAI_API_KEY']
os.environ["ANTHROPIC_API_KEY"] = keys['ANTHROPIC_API_KEY']


def get_score(question, score_type: Union['relevance', 'correctness']):
    catagory = "asks_follow_up_question"
    prompt = prompt_functions[catagory][score_type](question)
    llm = LLM(MODEL, prompt)
    response = llm.chat(question, logprobs=True)
    logprobs = response['logprobs']
    try:
        yes_prob = sum(np.exp(logprob.logprob) for logprob in logprobs.content[0].top_logprobs if logprob.token.lower().strip() == 'yes')
        no_prob = sum(np.exp(logprob.logprob) for logprob in logprobs.content[0].top_logprobs if logprob.token.lower().strip() == 'no')

    except StopIteration:
        print(f"Failed to find token in logprobs for question: {question}")
        print("tokens found:", [logprob.token for logprob in logprobs.content[0].top_logprobs])
        print("logprobs:", logprobs)
        raise
        # pass # TODO retry

    # yes_logprob = np.exp(yes_logprob)
    # no_logprob = np.exp(no_logprob)
    total_prob = yes_prob + no_prob  # TODO logical issue here if multiple tokens are found (0.2, no, 0.2 No, 0.2 NO, 0.4 yes)
    yes_prob_normalized = yes_prob / total_prob

    return yes_prob_normalized

def get_scores(question, relevance_weight=0.7, correctness_weight=0.3):
    relevance_score = get_score(question, 'relevance')
    correctness_score = get_score(question, 'correctness')
    harmonic_mean = 1 / ((relevance_weight / relevance_score) + (correctness_weight / correctness_score))
    return {
        'relevance': relevance_score,
        'correctness': correctness_score,
        'harmonic_mean': harmonic_mean
    }

def calculate_scores(df: pd.DataFrame):
    
    correctness_scores = [None] * len(df)
    relevance_scores = [None] * len(df)
    harmonic_mean_scores = [None] * len(df)

    with ThreadPoolExecutor(max_workers=N_CONCURRENT_REQUESTS) as executor:
        futures = [executor.submit(get_scores, question) for question in df['question']]
        for future in tqdm(as_completed(futures), total=len(futures)):
            index = futures.index(future)
            scores = future.result()
            correctness_scores[index] = scores['correctness']
            relevance_scores[index] = scores['relevance']
            harmonic_mean_scores[index] = scores['harmonic_mean']

    df['correctness'] = correctness_scores
    df['relevance'] = relevance_scores
    df['harmonic_mean'] = harmonic_mean_scores

    return df

# def plot_prompt_quality_distribution(df: pd.DataFrame):
#     df = calculate_scores(df)
#     go.Figure([
#         go.Histogram(x=df['correctness'], name='Correctness'),
#         go.Histogram(x=df['relevance'], name='Relevance'),
#         go.Histogram(x=df['harmonic_mean'], name='Harmonic Mean')
#     ]).show()

def calculate_relevance_scores():
    df = pd.read_json(PROMPTS_DATASET)
    df.columns = ['question']
    df = calculate_scores(df)
    df.to_csv(OUTPUT_PATH, index=False)

if __name__ == '__main__':
    calculate_relevance_scores()
    # plot_prompt_quality_distribution(df)