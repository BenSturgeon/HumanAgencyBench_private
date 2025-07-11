from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import threading

import pandas as pd
import google.generativeai.types as generation_types

from src.prompts.prompt_base import PromptBase
from src.llms import LLM
from src.utils import hash_cache, sum_deductions_from_json
from src.batch_utils import batch_model_response

N_CONCURRENT_REQUESTS = 1000


@hash_cache()
def model_response(prompt, system_prompt, model, temperature, top_p, max_tokens):
    # Removed semaphore acquisition logic for provider-specific throttling
    response = ""
    llm = LLM(model, system_prompt)
    try:
        response = llm.chat(prompt, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
    except generation_types.StopCandidateException as e:
        print(f"Warning: Gemini model '{model}' stopped due to RECITATION. Prompt: '{prompt[:100]}...' Exception: {e}")
        response = "[GENERATION STOPPED DUE TO RECITATION]"
    except Exception as e:
        print(f"Error during llm.chat for model {model}: {e}")
        status_code = getattr(getattr(e, 'response', None), 'status_code', None)
        if status_code == 429:
            response = "[RATE LIMIT ERROR DURING LLM CHAT]"
        elif status_code == 529:
            response = "[ANTHROPIC OVERLOAD ERROR DURING LLM CHAT]"
        else:
            response = "[ERROR DURING LLM CHAT]"
    return response, system_prompt


def get_model_responses(
    prompts, 
    system_prompts, model: str, temperature: float, top_p: float, max_tokens: int, use_cache: bool, refresh_cache: bool
):
    """Retrieve model responses for a list of prompts.

    If the target model is an Anthropic Claude model, we attempt to use the
    newly-added Message Batches API for efficiency.  If that fails for any
    reason, we transparently fall back to the existing per-prompt threaded
    logic so the evaluation never breaks.
    """

    # Try batched path using helper (works only when system prompts identical)
    if model.startswith("claude") and len(set(system_prompts)) == 1:
        try:
            responses = batch_model_response(
                prompts=prompts,
                system_prompt=system_prompts[0],
                model=model,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            return responses, system_prompts
        except Exception as e:
            print(f"[WARN] batch_model_response failed for model {model}: {e}. Falling back to per-prompt mode.")

    # Fallback: original multi-threaded per-prompt logic.
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
                refresh_cache=refresh_cache,
            ): i
            for i, (prompt, system_prompt) in enumerate(zip(prompts, system_prompts))
        }
        for future in tqdm(
            as_completed(future_to_index),
            total=len(future_to_index),
            desc=f"Getting student model responses: {model}",
        ):
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
        max_tokens=evaluator_max_tokens,
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
        

def get_scores(
    prompts, 
    subject_responses, 
    prompt_object, 
    evaluator_model, 
    use_cache, 
    refresh_cache, 
    subject_model, 
    misinformation=None,
    evaluator_max_tokens: int = 5000
):
    """Compute scores for a list of subject model responses.

    Uses Anthropic Message-Batch API when the evaluator model is a Claude
    variant, otherwise falls back to the existing threaded single-request
    logic.
    """

    # Try batch for Claude evaluator
    try:
        if evaluator_model.startswith("claude"):
            system_prompt = prompt_object.subject_evaluation_system_prompt()

            # Build evaluation prompts list (same logic as in score_response)
            eval_prompts_local = []
            for i, (prompt, response) in enumerate(zip(prompts, subject_responses)):
                if misinformation:
                    eval_prompts_local.append(
                        prompt_object.subject_evaluation_prompt(prompt, response, misinformation[i])
                    )
                else:
                    eval_prompts_local.append(
                        prompt_object.subject_evaluation_prompt(prompt, response)
                    )

            print(f"[DEBUG] Using batch API for evaluator model '{evaluator_model}' with {len(eval_prompts_local)} evaluation prompts")
            evaluator_responses = batch_model_response(
                prompts=eval_prompts_local,
                system_prompt=system_prompt,
                model=evaluator_model,
                temperature=0,
                top_p=1,
                max_tokens=evaluator_max_tokens,
            )
            print(f"[DEBUG] Received evaluator batch responses ({len(evaluator_responses)})")

            # Post-process: compute scores and deductions
            scores = []
            system_prompts = []
            eval_prompts_out = []
            for eval_prompt, evaluator_response in zip(eval_prompts_local, evaluator_responses):
                try:
                    total_deduction = sum_deductions_from_json(
                        evaluator_response, prompt_object.deductions
                    )
                except Exception:
                    # Mirror error handling of score_response
                    error_placeholders = [
                        "[RATE LIMIT ERROR DURING LLM CHAT]",
                        "[ERROR DURING LLM CHAT]",
                        "[GENERATION STOPPED DUE TO RECITATION]",
                        "[ANTHROPIC OVERLOAD ERROR DURING LLM CHAT]",
                    ]
                    if evaluator_response in error_placeholders:
                        total_deduction = prompt_object.top_eval_score
                    else:
                        total_deduction = prompt_object.top_eval_score

                score_val = max(prompt_object.top_eval_score - total_deduction, 0)
                scores.append(score_val)
                system_prompts.append(system_prompt)
                eval_prompts_out.append(eval_prompt)

            return (
                scores,
                system_prompts,
                eval_prompts_out,
                evaluator_responses,
            )
    except Exception as e:
        print(f"[WARN] Batch evaluator path failed for model {evaluator_model}: {e}. Falling back to per-prompt mode.")

    # Fallback: threaded per-prompt scoring
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
                evaluator_max_tokens=evaluator_max_tokens,
            ): i
            for i, (prompt, response) in enumerate(zip(prompts, subject_responses))
        }
        for future in tqdm(
            as_completed(future_to_index),
            total=len(future_to_index),
            desc=f"Scoring responses: {subject_model}",
        ):
            index = future_to_index[future]
            (
                scores[index],
                system_prompts[index],
                eval_prompts[index],
                evaluator_responses[index],
            ) = future.result()

    return scores, system_prompts, eval_prompts, evaluator_responses


def evaluate_model(
    prompts,
    evaluator_model,
    subject_model,
    subject_model_temperature,
    subject_model_top_p, subject_max_tokens, prompt_object,
    use_cache, refresh_cache,
    evaluator_max_tokens: int = 5000,
    gemini_max_tokens: int = 8192,
    misinformation=None
): # Note: gemini_max_tokens default isn't used if passed from config

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

    subject_responses, subject_system_prompts = get_model_responses(
        prompts=prompts, 
        system_prompts=subject_model_system_prompt, 
        model=subject_model, 
        temperature=subject_model_temperature, 
        top_p=subject_model_top_p,
        max_tokens=current_subject_max_tokens, 
        use_cache=use_cache, 
        refresh_cache=refresh_cache
    )
    
    scores, evaluator_system_prompts, evaluator_prompts, evaluator_responses = \
        get_scores(
            prompts=prompts,
            subject_responses=subject_responses,
            prompt_object=prompt_object,
            evaluator_model=evaluator_model, 
            use_cache=use_cache, 
            refresh_cache=refresh_cache, 
            subject_model=subject_model, 
            misinformation=misinformation,
            evaluator_max_tokens=evaluator_max_tokens
        )
    
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
    gemini_max_tokens: int = 8192,
    misinformation: List[str] = None,
    evaluator_max_tokens: int = 5000,
) -> pd.DataFrame:
    dfs = []

    with ThreadPoolExecutor(max_workers=len(subject_models)) as executor:
        futures_to_index = {}
        for subject_model in subject_models:
            futures_to_index.update({
                executor.submit(
                    evaluate_model,
                    prompts=prompts,
                    evaluator_model=evaluator_model,
                    subject_model=subject_model,
                    subject_model_temperature=subject_model_temperature,
                    subject_model_top_p=subject_model_top_p,
                    subject_max_tokens=subject_max_tokens,
                    prompt_object=prompt_object,
                    use_cache=use_cache,
                    refresh_cache=refresh_cache,
                    gemini_max_tokens=gemini_max_tokens,
                    misinformation=misinformation,
                    evaluator_max_tokens=evaluator_max_tokens,
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
