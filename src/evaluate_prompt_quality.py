from typing import Union, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
import numpy as np
import json
import pandas as pd

from src.prompts import prompt_objects, PromptBase
from src.llms import LLM
from src.utils import hash_cache


N_CONCURRENT_REQUESTS = 200


@hash_cache()
def get_scores(prompt, model, problem_type):

    prompt = prompt_objects[problem_type]().relevance_check(prompt)
    system_prompt = "You are an evaluator. Give a score based on the provided criteria. Output only the integer score."
    llm = LLM(model, system_prompt)
    response = llm.chat(prompt, temperature=0)

    relevance_score = None
    for word in response.split():
        try:
            num = int(word)
            if 0 <= num <= 1000:
                relevance_score = num
                break
        except ValueError:
            continue

    if relevance_score is None:
        raise ValueError(f"Prompt relevance score must be between 0 and 1000. Got {score}")

    return relevance_score, system_prompt, prompt


def calculate_prompt_scores(
        prompts, 
        model, 
        problem_type, 
        n_relevant_prompts,
        use_cache, 
        refresh_cache
) -> Dict[str, list]:

    relevance_scores = [None] * len(prompts)
    relevance_system_prompts = [None] * len(prompts)
    relevance_prompts = [None] * len(prompts)

    with ThreadPoolExecutor(max_workers=N_CONCURRENT_REQUESTS) as executor:

        future_to_index = {executor.submit(
            get_scores, 
            prompt=prompt,
            problem_type=problem_type,
            model=model,
            use_cache=use_cache,
            refresh_cache=refresh_cache
            ): i for i, prompt in enumerate(prompts)
        }
        for future in tqdm(as_completed(future_to_index), total=len(future_to_index), desc='Scoring prompts'):

            index = future_to_index[future]
            relevance_scores[index], relevance_system_prompts[index], relevance_prompts[index] = future.result()

        # get indices of top n prompts
        relevance_scores = np.array(relevance_scores)
        top_n_indices = np.argsort(relevance_scores)[-n_relevant_prompts:]
        passed_evaluation = np.zeros(len(prompts), dtype=bool)
        passed_evaluation[top_n_indices] = True

    return relevance_scores, relevance_system_prompts, relevance_prompts, passed_evaluation
