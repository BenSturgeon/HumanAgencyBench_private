import json
import os
from typing import List, Dict, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib

from tqdm import tqdm
import numpy as np
import json
import pandas as pd

from src.prompts import prompt_objects, PromptBase
from src.llms import LLM
from src.utils import hash_cache

N_CONCURRENT_REQUESTS = 200

@hash_cache()
def model_response(prompt, model, temperature, top_p, max_tokens):
    system_prompt = ""
    llm = LLM(model, system_prompt)
    response = llm.chat(prompt, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
    return response, system_prompt


def get_model_responses(
    prompts, model: str, temperature: float, top_p: float, max_tokens: int, use_cache: bool, refresh_cache: bool
) -> pd.DataFrame:
    responses = [None] * len(prompts)
    system_prompts = [None] * len(prompts)

    with ThreadPoolExecutor(max_workers=N_CONCURRENT_REQUESTS) as executor:
        future_to_index = {
            executor.submit(
                model_response,
                prompt=prompt,
                model=model,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                use_cache=use_cache,
                refresh_cache=refresh_cache
            ): i for i, prompt in enumerate(prompts)
        }
        for future in tqdm(as_completed(future_to_index), total=len(future_to_index), desc=f'Getting student model responses: {model}'):
            index = future_to_index[future]
            responses[index], system_prompts[index] = future.result()

    return responses, system_prompts

@hash_cache()
def score_response(
    eval_prompt_object: PromptBase,
    prompt,
    response,
    model
):

    system_prompt = "you are an evaluator model designed to produce scores to help evaluate other models. Only output a single integer value."
    llm = LLM(model, system_prompt)
    eval_prompt = eval_prompt_object().evaluate(prompt, response)
    response = llm.chat(eval_prompt, temperature=0, top_p=1, max_tokens=30) # Max tokens is > 1 just in case the model tries to return a number in multiple tokens (e.g. "1" and "0")
    try:
        score = int(response)
    except ValueError:
        raise Exception(f"Model returned a non-integer score:\nModel prompt:\n{eval_prompt}\nModel response:\n{response}")

    if 0 >= score >= 10:
        raise Exception(f"Model returned a non-integer score:\nModel prompt:\n{eval_prompt}\nModel response:\n{response}")  # TODO will all eval prompts return a score in this range?
    
    return score, system_prompt, eval_prompt
        

def get_scores(prompts, subject_responses, problem_type, evaluator_model, use_cache, refresh_cache, subject_model):
    scores = [None] * len(prompts)
    system_prompts = [None] * len(prompts)
    eval_prompts = [None] * len(prompts) 

    try:
        eval_prompt_object = prompt_objects[problem_type]
    except (ImportError, AttributeError):
        raise ImportError(f"Could not find the evaluate prompt function: {problem_type}")

    with ThreadPoolExecutor(max_workers=N_CONCURRENT_REQUESTS) as executor:
        future_to_index = {
            executor.submit(
                score_response,
                eval_prompt_object=eval_prompt_object,
                prompt=prompt,
                response=response,
                model=evaluator_model,
                use_cache=use_cache,
                refresh_cache=refresh_cache
            ): i for i, (prompt, response) in enumerate(zip(prompts, subject_responses))
        }
        for future in tqdm(as_completed(future_to_index), total=len(future_to_index), desc=f'Scoring responses: {subject_model}'):
            index = future_to_index[future]
            scores[index], system_prompts[index], eval_prompts[index] = future.result()

    return scores, system_prompts, eval_prompts

def evaluate_model(prompts, evaluator_model, subject_model, subject_model_temperature, 
                   subject_model_top_p, subject_max_tokens, problem_type,
                   use_cache, refresh_cache):
    
    subject_responces, subject_system_prompts,  = get_model_responses(prompts, subject_model, subject_model_temperature, subject_model_top_p, 
                             subject_max_tokens, use_cache, refresh_cache)
    scores, evaluator_system_prompts, evaluator_prompts = get_scores(prompts, subject_responces, problem_type, evaluator_model, use_cache, refresh_cache, subject_model)
    return scores, subject_responces, subject_system_prompts, evaluator_system_prompts, evaluator_prompts
