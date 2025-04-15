from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import pandas as pd

from src.prompts.prompt_base import PromptBase
from src.llms import LLM
from src.utils import hash_cache, sum_deductions_from_json

N_CONCURRENT_REQUESTS = 200


@hash_cache()
def model_response(prompt, system_prompt, model, temperature, top_p, max_tokens):
    llm = LLM(model, system_prompt)
    response = llm.chat(prompt, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
    return response, system_prompt


def get_model_responses(
    prompts, system_prompts, model: str, temperature: float, top_p: float, max_tokens: int, use_cache: bool, refresh_cache: bool
):
    responses = [None] * len(prompts)

    with ThreadPoolExecutor(max_workers=N_CONCURRENT_REQUESTS) as executor:
        future_to_index = {
            executor.submit(
                model_response,
                prompt=prompt,
                system_prompt=system_prompt,
                model=model,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                use_cache=use_cache,
                refresh_cache=refresh_cache
            ): i for i, (prompt, system_prompt) in enumerate(zip(prompts, system_prompts))
        }
        for future in tqdm(as_completed(future_to_index), total=len(future_to_index), desc=f'Getting student model responses: {model}'):
            index = future_to_index[future]
            responses[index], system_prompts[index] = future.result()

    return responses, system_prompts


def score_response(
    prompt_object: PromptBase,
    prompt,
    response,
    model,
    use_cache: bool,
    refresh_cache: bool,
    misinformation=None
):
    system_prompt = prompt_object.subject_evaluation_system_prompt()
    
    if misinformation:
        eval_prompt = prompt_object.subject_evaluation_prompt(prompt, response, misinformation)
    else:
        eval_prompt = prompt_object.subject_evaluation_prompt(prompt, response)

    response = model_response(
        prompt=eval_prompt,
        system_prompt=system_prompt,
        model=model,
        temperature=0,
        top_p=1,
        max_tokens=200,
        use_cache=use_cache,
        refresh_cache=refresh_cache
    )[0]

    total_deduction = sum_deductions_from_json(response, prompt_object.deductions)
    score = max(prompt_object.top_eval_score - total_deduction, 0)

    if not (-1 <= score <= prompt_object.top_eval_score):
        raise Exception(f"Model returned a score out of bounds. Score: {score}, Top possible score: {prompt_object.top_eval_score}") 
    
    return score, system_prompt, eval_prompt, response
        

def get_scores(prompts, subject_responses, prompt_object, evaluator_model, use_cache, refresh_cache, subject_model, misinformation = None):

    scores = [None] * len(prompts)
    system_prompts = [None] * len(prompts)
    eval_prompts = [None] * len(prompts) 
    evaluator_responses = [None] * len(prompts) 

    with ThreadPoolExecutor(max_workers=N_CONCURRENT_REQUESTS) as executor:
        future_to_index = {
            executor.submit(
                score_response,
                prompt_object=prompt_object,
                prompt=prompt,
                response=response,
                model=evaluator_model,
                use_cache=use_cache,
                refresh_cache=refresh_cache,
                misinformation=misinformation[i] if misinformation else None
            ): i for i, (prompt, response) in enumerate(zip(prompts, subject_responses))
        }
        for future in tqdm(as_completed(future_to_index), total=len(future_to_index), desc=f'Scoring responses: {subject_model}'):
            index = future_to_index[future]
            scores[index], system_prompts[index], eval_prompts[index] , evaluator_responses[index]= future.result()

    return scores, system_prompts, eval_prompts, evaluator_responses


def evaluate_model(prompts, evaluator_model, subject_model, subject_model_temperature, 
                   subject_model_top_p, subject_max_tokens, prompt_object,
                   use_cache, refresh_cache, misinformation=None):

    subject_model_system_prompt = [prompt_object.subject_model_system_prompt() for _ in range(len(prompts))]

    subject_responses, subject_system_prompts = get_model_responses(prompts, subject_model_system_prompt, subject_model, subject_model_temperature, subject_model_top_p, 
                             subject_max_tokens, use_cache, refresh_cache)
    
    scores, evaluator_system_prompts, evaluator_prompts, evaluator_responses = \
        get_scores(prompts, subject_responses, prompt_object, evaluator_model, use_cache, refresh_cache, subject_model, misinformation)
    
    return scores, subject_responses, subject_system_prompts, evaluator_system_prompts, evaluator_prompts, evaluator_responses


def evaluate_many_subject_models(
    prompts: List[str],
    subject_models: List[str],
    evaluator_model: str,
    subject_model_temperature: float,
    subject_model_top_p: float,
    subject_max_tokens: int,
    prompt_object: PromptBase,
    use_cache: bool,
    refresh_cache: bool,
    misinformation: List[str] = None
) -> pd.DataFrame:
    dfs = []

    with ThreadPoolExecutor(max_workers=len(subject_models)) as executor:
        futures_to_index = {}
        for subject_model in subject_models:
            futures_to_index.update({
                executor.submit(
                    evaluate_model,
                    prompts,
                    evaluator_model,
                    subject_model,
                    subject_model_temperature,
                    subject_model_top_p,
                    subject_max_tokens,
                    prompt_object,
                    use_cache,
                    refresh_cache,
                    misinformation
                ): subject_model
            })

        for future in as_completed(futures_to_index):
            subject_model = futures_to_index[future]
            scores, subject_responses, subject_system_prompts, evaluator_system_prompts, evaluator_prompts, evaluator_responses = future.result()

            df = pd.DataFrame({
                'prompt': prompts,
                'score': scores,
                'subject_response': subject_responses,
                'subject_system_prompt': subject_system_prompts,
                'evaluator_system_prompt': evaluator_system_prompts,
                'evaluator_prompt': evaluator_prompts,
                'evaluator_response': evaluator_responses,
                'subject_model': subject_model
            })

            dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    return df
