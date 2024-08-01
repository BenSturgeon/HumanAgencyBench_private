from typing import List, Union
import os
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
import pandas as pd

from src.prompts import threatening_message_if_not_json, AsksFollowUp, CorrectMisinformation, CitesSources
from src.llms import LLM
from src.utils import hash_cache

HUMAN_EXPERT_FILE = "human_expert_prompts.csv"
N_CONCURRENT_REQUESTS = 200
DATASETS = {
    "asks_follow_up_question": AsksFollowUp,
    "correct_misinformation": CorrectMisinformation,
    "cites_sources": CitesSources
}

def generate_dataset(
    subdim: str,
    model: str = "gpt-4o",
    n_prompts: int = 1000,
    temperature: float = 1.0,
    max_tokens: int = 500,
    top_p: float = 0.9,
    n_examples_shown_per_generation: int = 5,
    n_prompts_per_generation: int = 10,
    use_cache: bool = True,
    refresh_cache: bool = False,
) -> Union[List[str], List[str], List[str]]:

    requests_needed = n_prompts // n_prompts_per_generation
    generated_prompts = [None] * n_prompts
    system_prompts = [None] * n_prompts
    generative_prompts = [None] * n_prompts

    try:
        generate = DATASETS[subdim].generate
    except (ImportError, AttributeError):
        raise ImportError(f"Could not find the generation prompt function: {generation_prompt}")

    @hash_cache()
    def generate_single_prompt(
        model: str, 
        generation_function: callable,
        example_prompts: list, 
        max_retries=5):
        """
        example_prompts: list of example prompts to be used as reference
        i: index of the request. Used for caching
        """
        system_prompt = ""

        # It's possible that max retries will be reached but just keeping it here to make sure we don't get stuck in an infinite loop
        for _ in range(max_retries):  
            llm = LLM(model, system_prompt)
            generative_prompt = generation_function()
            generative_prompt += f"\n{threatening_message_if_not_json()}"
            response = llm.chat(prompt=generative_prompt, temperature=temperature, max_tokens=max_tokens, top_p=top_p)

            try:
                generated_prompts = json.loads(response)  # the generation prompts asks for 10 prompts as json
            except json.JSONDecodeError:
                continue

            generated_prompts = list(generated_prompts.values())
            if len(generated_prompts) != 10:
                continue

            return generated_prompts, system_prompt, generative_prompt
        
        raise Exception(f"Failed to generate prompts after {max_retries} retries: \nLast response: {response}")

    with ThreadPoolExecutor(max_workers=N_CONCURRENT_REQUESTS) as executor:
        future_to_index = {executor.submit(
            generate_single_prompt, 
            model=model,
            generation_prompt_func=generation_prompt_func,
            example_prompts=human_expert_example_prompts,
            cache_nonce=cache_nonce,
            use_cache=use_cache,
            refresh_cache=refresh_cache
            ): cache_nonce for cache_nonce in range(requests_needed)
        }
        for future in tqdm(as_completed(future_to_index), total=requests_needed, desc='Generating prompts'):

            generated_prompt, system_prompt, generative_prompt = future.result()

            index = future_to_index[future]
            generated_prompts[index * n_prompts_per_generation: (index + 1) * n_prompts_per_generation] = generated_prompt
            system_prompts[index * n_prompts_per_generation: (index + 1) * n_prompts_per_generation] = [system_prompt] * n_prompts_per_generation
            generative_prompts[index * n_prompts_per_generation: (index + 1) * n_prompts_per_generation] = [generative_prompt] * n_prompts_per_generation

    return generated_prompts, system_prompts, generative_prompts
