from typing import List, Union
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

from tqdm import tqdm

from src.prompts.prompt_base import PromptBase
from src.llms import LLM
from src.utils import hash_cache
from src.prompts.prompts import prompt_objects

N_CONCURRENT_REQUESTS = 1000


@hash_cache()
def generate_single_prompt(
    model: str, 
    generative_prompt: str,
    system_prompt: str,
    n_prompts_created_per_generation: int,
    temperature: float,
    max_tokens: int,
    top_p: float,
    require_misinformation_fields: bool = False,
    max_retries: int = 10
):
    """
    example_prompts: list of example prompts to be used as reference
    i: index of the request. Used for caching
    """

    # It's possible that max retries will be reached but just keeping it here to make sure we don't get stuck in an infinite loop
    for _ in range(max_retries):  
        llm = LLM(model, system_prompt=system_prompt)

        response = llm.chat(prompt=generative_prompt, temperature=temperature, max_tokens=max_tokens, top_p=top_p, return_json=True)

        # Clean potential JSON formatting from the response
        if "```json" in response:
            response = response.replace("```json", "").replace("```", "").strip()
        elif "```" in response: # Handle cases where only ``` is present
            response = response.replace("```", "").strip()

        try:
            prompt = json.loads(response)
            if isinstance(prompt, dict):
                generated_subject_prompts = list(prompt.values())
            elif isinstance(prompt, list):
                generated_subject_prompts = prompt
            else:
                continue  # Retry if it's neither a dict nor a list
        except json.JSONDecodeError:
            continue  # Retry if the response is not JSON

        # Basic length check
        if len(generated_subject_prompts) != n_prompts_created_per_generation:
            continue  # Retry if the number of prompts generated is not as expected

        # Additional validation for correct_misinformation style prompts
        if require_misinformation_fields:
            if not all(isinstance(item, dict) and 'paragraph' in item and 'misinformation' in item for item in generated_subject_prompts):
                # Retry if any prompt is missing required keys
                continue

        return generated_subject_prompts, system_prompt, generative_prompt
    
    raise Exception(f"Failed to generate prompts after {max_retries} retries: \nLast response: {response}")


def generate_dataset(
    model: str,
    n_prompts: int,
    temperature: float,
    max_tokens: int,
    top_p: float,
    prompt_object: PromptBase,
    n_prompts_created_per_generation: int,
    use_cache: bool = True,
    refresh_cache: bool = False,
) -> Union[List[str], List[str], List[str]]:

    requests_needed = (n_prompts + n_prompts_created_per_generation - 1) // n_prompts_created_per_generation
    generated_prompts = [None] * n_prompts

    random.seed(42)  # This is needed because the prompt generation uses random so we need to make it deterministic to allow caching to work
    system_prompts = [prompt_object.generative_system_prompt() for _ in range(n_prompts)]
    generative_prompts = [prompt_object.generative_prompt(n_prompts_created_per_generation) for _ in range(n_prompts)]

    if model.startswith("claude"):
        # Use batch API & caching helper
        from src.batch_utils import batch_model_response

        print(f"[DEBUG] Batch generating {requests_needed} prompt groups using Claude batch API")

        batched_responses = batch_model_response(
            prompts=generative_prompts[:requests_needed],
            system_prompt=system_prompts[0],
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        # Parse each JSON response using existing validation logic
        import json
        for idx, raw in enumerate(batched_responses):
            resp = raw
            if "```json" in resp:
                resp = resp.replace("```json", "").replace("```", "").strip()
            elif "```" in resp:
                resp = resp.replace("```", "").strip()

            try:
                obj = json.loads(resp)
                if isinstance(obj, dict):
                    parsed_list = list(obj.values())
                elif isinstance(obj, list):
                    parsed_list = obj
                else:
                    raise ValueError
            except Exception:
                # Fallback: re-query this prompt via single call
                print("[WARN] Batch generation item unparsable – retrying individually.")
                parsed_list, _, _ = generate_single_prompt(
                    model=model,
                    generative_prompt=generative_prompts[idx],
                    system_prompt=system_prompts[idx],
                    n_prompts_created_per_generation=n_prompts_created_per_generation,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    require_misinformation_fields=isinstance(prompt_object, prompt_objects['correct_misinformation']),
                    use_cache=False,
                    refresh_cache=False,
                )

            if len(parsed_list) != n_prompts_created_per_generation:
                raise Exception("Batch generation returned incorrect number of prompts")

            start_idx = idx * n_prompts_created_per_generation
            end_idx = start_idx + n_prompts_created_per_generation
            generated_prompts[start_idx:end_idx] = parsed_list
            system_prompts[start_idx:end_idx] = [system_prompts[idx]] * n_prompts_created_per_generation
            generative_prompts[start_idx:end_idx] = [generative_prompts[idx]] * n_prompts_created_per_generation

    else:
        # Original threaded path
        with ThreadPoolExecutor(max_workers=N_CONCURRENT_REQUESTS) as executor:
            future_to_index = {executor.submit(
                generate_single_prompt,
                model=model,
                generative_prompt=generative_prompts[i],
                system_prompt=system_prompts[i],
                n_prompts_created_per_generation=n_prompts_created_per_generation,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                require_misinformation_fields=isinstance(prompt_object, prompt_objects['correct_misinformation']),
                cache_nonce=i,
                use_cache=use_cache,
                refresh_cache=refresh_cache
                ): i for i in range(requests_needed)
            }
            for future in tqdm(as_completed(future_to_index), total=requests_needed, desc='Generating prompts'):

                generated_prompt, system_prompt, generative_prompt = future.result()
                index = future_to_index[future]
                start_idx = index * n_prompts_created_per_generation
                end_idx = (index + 1) * n_prompts_created_per_generation

                generated_prompts[start_idx: end_idx] = generated_prompt
                system_prompts[start_idx: end_idx] = [system_prompt] * n_prompts_created_per_generation
                generative_prompts[start_idx: end_idx] = [generative_prompt] * n_prompts_created_per_generation

    return generated_prompts, system_prompts, generative_prompts
