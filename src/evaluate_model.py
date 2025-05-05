from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import threading

import pandas as pd
import google.generativeai.types as generation_types

from src.prompts.prompt_base import PromptBase
from src.llms import LLM
from src.utils import hash_cache, sum_deductions_from_json

N_CONCURRENT_REQUESTS = 1000
MAX_GEMINI_WORKERS = 5 # Limit concurrent Gemini initializations/calls
gemini_semaphore = threading.Semaphore(MAX_GEMINI_WORKERS)

# General semaphore for most Anthropic models
MAX_ANTHROPIC_WORKERS = 5 # Limit concurrent Anthropic calls (tune as needed)
anthropic_semaphore = threading.Semaphore(MAX_ANTHROPIC_WORKERS)

# Specific, stricter semaphore for Claude 3.7 Sonnet
MAX_CLAUDE_3_7_WORKERS = 2 # Limit concurrent 3.7 Sonnet calls (tune as needed)
claude_3_7_semaphore = threading.Semaphore(MAX_CLAUDE_3_7_WORKERS)


@hash_cache()
def model_response(prompt, system_prompt, model, temperature, top_p, max_tokens):
    is_gemini = model.startswith("models/gemini")
    is_claude_3_7 = model == "claude-3-7-sonnet-20250219"
    is_other_claude = model.startswith("claude-") and not is_claude_3_7
    
    semaphore_acquired = None # Use None to indicate no semaphore acquired initially
    try:
        if is_gemini:
            gemini_semaphore.acquire()
            semaphore_acquired = 'gemini'
        elif is_claude_3_7:
            claude_3_7_semaphore.acquire() # Acquire specific semaphore
            semaphore_acquired = 'claude_3_7'
        elif is_other_claude: # Use the general check for other Claude models
            anthropic_semaphore.acquire() # Use the general semaphore
            semaphore_acquired = 'anthropic' # Indicate general anthropic semaphore
            
        response = "" # Default response in case of error
        llm = LLM(model, system_prompt)
        try:
            response = llm.chat(prompt, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        except generation_types.StopCandidateException as e:
            print(f"Warning: Gemini model '{model}' stopped due to RECITATION. Prompt: '{prompt[:100]}...' Exception: {e}")
            response = "[GENERATION STOPPED DUE TO RECITATION]" 
        except Exception as e:
            print(f"Error during llm.chat for model {model}: {e}")
            # Check if the exception has a response attribute with status_code (common for httpx errors)
            status_code = getattr(getattr(e, 'response', None), 'status_code', None)
            if status_code == 429:
                 response = "[RATE LIMIT ERROR DURING LLM CHAT]"
            # Add check for Anthropic overload error (HTTP 529)
            elif status_code == 529:
                 response = "[ANTHROPIC OVERLOAD ERROR DURING LLM CHAT]"
            else:
                 response = "[ERROR DURING LLM CHAT]" 
        return response, system_prompt
    finally:
        if semaphore_acquired == 'gemini':
            gemini_semaphore.release()
        elif semaphore_acquired == 'claude_3_7':
             claude_3_7_semaphore.release() # Release specific semaphore
        elif semaphore_acquired == 'anthropic': # Use the general indicator
            anthropic_semaphore.release() # Release the general semaphore


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
    misinformation=None,
    evaluator_max_tokens: int = 5000
):
    system_prompt = prompt_object.subject_evaluation_system_prompt()
    
    if misinformation:
        eval_prompt = prompt_object.subject_evaluation_prompt(prompt, response, misinformation)
    else:
        eval_prompt = prompt_object.subject_evaluation_prompt(prompt, response)

    # Call model_response with the evaluator_max_tokens parameter
    evaluator_response_text = model_response(
        prompt=eval_prompt,
        system_prompt=system_prompt,
        model=model,
        temperature=0,
        top_p=1,
        max_tokens=evaluator_max_tokens, # Use the parameter here
        use_cache=use_cache,
        refresh_cache=refresh_cache
    )[0]

    # Check for known error placeholders before attempting JSON parsing
    error_placeholders = [
        "[RATE LIMIT ERROR DURING LLM CHAT]",
        "[ERROR DURING LLM CHAT]",
        "[GENERATION STOPPED DUE TO RECITATION]",
        "[ANTHROPIC OVERLOAD ERROR DURING LLM CHAT]"
    ]

    if evaluator_response_text in error_placeholders:
        print(f"Warning: Evaluator model '{model}' returned an error placeholder: {evaluator_response_text}. Assigning score 0.")
        # Assign max deduction so score becomes 0 after max(..., 0)
        total_deduction = prompt_object.top_eval_score 
    else:
        try:
            total_deduction = sum_deductions_from_json(evaluator_response_text, prompt_object.deductions)
        except ValueError as e: # Catch parsing errors specifically
            print(f"Error parsing JSON from evaluator '{model}'. Assigning score 0. Error: {e}")
            print(f"Evaluator Response causing error: {evaluator_response_text}")
            total_deduction = prompt_object.top_eval_score # Assign max deduction

    score = max(prompt_object.top_eval_score - total_deduction, 0)

    if not (0 <= score <= prompt_object.top_eval_score): # Adjusted lower bound check
        print(f"Warning: Score calculation resulted in unexpected value. Score: {score}, Top possible score: {prompt_object.top_eval_score}, Deduction: {total_deduction}. Clamping score.")
        score = max(0, min(score, prompt_object.top_eval_score))
    
    # Return the original evaluator response text, even if it was an error
    return score, system_prompt, eval_prompt, evaluator_response_text
        

def get_scores(prompts, subject_responses, prompt_object, evaluator_model, use_cache, refresh_cache, subject_model, misinformation = None,
               evaluator_max_tokens: int = 5000):

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
                misinformation=misinformation[i] if misinformation else None,
                evaluator_max_tokens=evaluator_max_tokens
            ): i for i, (prompt, response) in enumerate(zip(prompts, subject_responses))
        }
        for future in tqdm(as_completed(future_to_index), total=len(future_to_index), desc=f'Scoring responses: {subject_model}'):
            index = future_to_index[future]
            scores[index], system_prompts[index], eval_prompts[index] , evaluator_responses[index]= future.result()

    return scores, system_prompts, eval_prompts, evaluator_responses


def evaluate_model(prompts, evaluator_model, subject_model, subject_model_temperature, 
                   subject_model_top_p, subject_max_tokens, prompt_object,
                   use_cache, refresh_cache, misinformation=None,
                   evaluator_max_tokens: int = 5000, 
                   gemini_max_tokens: int = 3000): # Note: gemini_max_tokens default isn't used if passed from config

    subject_model_system_prompt = [prompt_object.subject_model_system_prompt() for _ in range(len(prompts))]

    # Select the appropriate max_tokens based on the specific subject model
    specific_gemini_models = [
        "models/gemini-2.5-pro-preview-03-25",
        "models/gemini-2.5-flash-preview-04-17"
    ]
    
    # Apply gemini_max_tokens if the model is one of the specific Gemini models
    if subject_model in specific_gemini_models:
        current_subject_max_tokens = gemini_max_tokens
        print(f"Using gemini_max_tokens ({gemini_max_tokens}) for {subject_model}") # Optional: logging
    else:
        current_subject_max_tokens = subject_max_tokens

    subject_responses, subject_system_prompts = get_model_responses(prompts, subject_model_system_prompt, subject_model, subject_model_temperature, subject_model_top_p, 
                             current_subject_max_tokens, use_cache, refresh_cache) 
    
    scores, evaluator_system_prompts, evaluator_prompts, evaluator_responses = \
        get_scores(prompts, subject_responses, prompt_object, evaluator_model, use_cache, refresh_cache, subject_model, misinformation,
                   evaluator_max_tokens=evaluator_max_tokens)
    
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
    misinformation: List[str] = None,
    evaluator_max_tokens: int = 5000,
    gemini_max_tokens: int = 3000
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
                    misinformation,
                    evaluator_max_tokens=evaluator_max_tokens,
                    gemini_max_tokens=gemini_max_tokens
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
