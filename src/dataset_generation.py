from typing import List, Union
import os
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
import pandas as pd

from src.prompts import threatening_message_if_not_json, prompt_functions
from src.llms import LLM
from src.utils import hash_cache

HUMAN_EXPERT_FILE = "human_expert_prompts.csv"
N_CONCURRENT_REQUESTS = 200


def generate_dataset(
    folder: str,
    model: str,
    n_prompts: int,
    generation_prompt: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    n_examples_shown_per_generation: int = 5,
    n_prompts_per_generation: int = 10,
    use_cache: bool = True,
    refresh_cache: bool = False,
    return_json= True
) -> Union[List[str], List[str], List[str]]:

    # load example prompts
    example_prompts_path = os.path.join(folder, HUMAN_EXPERT_FILE)
    human_expert_df = pd.read_csv(example_prompts_path)
    human_expert_example_prompts = human_expert_df['human_expert_prompt'].tolist()

    requests_needed = n_prompts // n_prompts_per_generation
    generated_prompts = [None] * n_prompts
    system_prompts = [None] * n_prompts
    generative_prompts = [None] * n_prompts

    try:
        generation_prompt_func = prompt_functions[generation_prompt]["generate"]
    except (ImportError, AttributeError):
        raise ImportError(f"Could not find the generation prompt function: {generation_prompt}")

    @hash_cache()
    def generate_single_prompt(
        model: str, 
        generation_prompt_func: callable,
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
            generative_prompt = generation_prompt_func(random.sample(example_prompts, n_examples_shown_per_generation))
            # generative_prompt += f"\n{threatening_message_if_not_json()}"
            response = llm.chat(prompt=generative_prompt, temperature=temperature, max_tokens=max_tokens, top_p=top_p, return_json=return_json)

            try:
                prompt = json.loads(response)
                if isinstance(prompt, dict):
                    generated_prompts = list(prompt.values())
                elif isinstance(prompt, list):
                    generated_prompts = prompt
                else:
                    continue  # Skip if it's neither a dict nor a list
            except json.JSONDecodeError:
                continue

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
