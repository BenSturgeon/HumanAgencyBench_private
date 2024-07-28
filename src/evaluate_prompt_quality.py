from typing import Union, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
import numpy as np
import json
import pandas as pd

from src.prompts import prompt_functions
from src.llms import LLM
from src.utils import hash_cache


N_CONCURRENT_REQUESTS = 200

def get_score(prompt, model, catagory, score_type: Union['relevance', 'correctness']):
    prompt = prompt_functions[catagory][score_type](prompt)
    system_prompt = ""
    llm = LLM(model, system_prompt)
    response = llm.chat(prompt, return_logprobs=True, max_tokens=1, temperature=0, top_p=1)
    logprobs = response['logprobs']
    try:
        yes_prob = sum(np.exp(logprob.logprob) for logprob in logprobs.content[0].top_logprobs if logprob.token.lower().strip() == 'yes')
        no_prob = sum(np.exp(logprob.logprob) for logprob in logprobs.content[0].top_logprobs if logprob.token.lower().strip() == 'no')

    except StopIteration:
        # This has not happend yet with larger models. So leaving it as is for now.
        print(f"Failed to find token in logprobs for question: {prompt}")
        print("tokens found:", [logprob.token for logprob in logprobs.content[0].top_logprobs])
        print("logprobs:", logprobs)
        raise

    total_prob = yes_prob + no_prob  
    yes_prob_normalized = yes_prob / total_prob

    return yes_prob_normalized, system_prompt, prompt

@hash_cache()
def get_scores(prompt, model, catagory, relevance_correctness_weight):

    
    relevance_score, relevenace_system_prompt,  relevenace_prompt, = get_score(prompt, model, catagory, 'relevance')
    correctness_score, correctness_system_prompt, correctness_prompt = get_score(prompt, model, catagory, 'correctness')
    harmonic_mean = 1 / ((relevance_correctness_weight / relevance_score) + ((1 - relevance_correctness_weight) / correctness_score))

    return relevance_score, correctness_score, harmonic_mean, relevenace_system_prompt, correctness_system_prompt, relevenace_prompt, correctness_prompt

def calculate_prompt_scores(
        prompts: str, model, catagory, hmean_threshold, relevance_correctness_weight, 
        use_cache, refresh_cache) -> Dict[str, list]:

    relevance_scores = [None] * len(prompts)
    correctness_scores = [None] * len(prompts)
    harmonic_mean_scores = [None] * len(prompts)
    relevance_system_prompts = [None] * len(prompts)
    correctness_system_prompts = [None] * len(prompts)
    relevance_prompts = [None] * len(prompts)
    correctness_prompts = [None] * len(prompts)


    with ThreadPoolExecutor(max_workers=N_CONCURRENT_REQUESTS) as executor:

        future_to_index = {executor.submit(
            get_scores, 
            prompt=prompt,
            catagory=catagory,
            model=model,
            relevance_correctness_weight=relevance_correctness_weight,
            use_cache=use_cache,
            refresh_cache=refresh_cache
            ): i for i, prompt in enumerate(prompts)
        }
        for future in tqdm(as_completed(future_to_index), total=len(future_to_index), desc='Scoring prompts'):

            index = future_to_index[future]
            relevance_scores[index], correctness_scores[index], harmonic_mean_scores[index], relevance_system_prompts[index], \
                correctness_system_prompts[index], relevance_prompts[index], correctness_prompts[index] = future.result()
            
    passed_evaluation = [h_mean > hmean_threshold for h_mean in harmonic_mean_scores]

    return correctness_scores, relevance_scores, harmonic_mean_scores, passed_evaluation, relevance_system_prompts, correctness_system_prompts, relevance_prompts, correctness_prompts
