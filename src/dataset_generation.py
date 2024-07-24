from typing import List
import os
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib

from tqdm import tqdm
import pandas as pd

from src.prompts import threatening_message_if_not_json, prompt_functions
from src.llms import LLM
from src.utils import hash_cache, setup_keys

HUMAN_EXPERT_FILE = "human_expert_prompts.csv"
N_CONCURRENT_REQUESTS = 100
RATE_LIMIT_RPM = 500


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
) -> List[str]: 

    # load example prompts
    example_prompts_path = os.path.join(folder, HUMAN_EXPERT_FILE)
    human_expert_df = pd.read_csv(example_prompts_path)
    human_expert_example_prompts = human_expert_df['human_expert_prompt'].tolist()

    requests_needed = n_prompts // n_prompts_per_generation
    request_interval = 60 / RATE_LIMIT_RPM
    last_request_time = time.time()
    generated_prompts = []

    try:
        module = importlib.import_module(f"src.prompts")
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

        # It's possible that max retries will be reached but just keeping it here to make sure we don't get stuck in an infinite loop
        for _ in range(max_retries):  
            llm = LLM(model, "")
            generative_prompt = generation_prompt_func(random.sample(example_prompts, n_examples_shown_per_generation))
            generative_prompt += f"\n{threatening_message_if_not_json()}"
            response = llm.chat(prompt=generative_prompt, temperature=temperature, max_tokens=max_tokens, top_p=top_p)

            try:
                generated_prompts = json.loads(response)  # the generation prompts asks for 10 prompts as json
            except json.JSONDecodeError:
                continue

            generated_prompts = list(generated_prompts.values())
            if len(generated_prompts) != 10:
                continue

            return generated_prompts
        
        raise Exception(f"Failed to generate prompts after {max_retries} retries: \n {response}")

    with ThreadPoolExecutor(max_workers=N_CONCURRENT_REQUESTS) as executor:
        futures = [
            executor.submit(
                generate_single_prompt, 
                model=model,
                generation_prompt_func=generation_prompt_func,
                example_prompts=human_expert_example_prompts,
                cache_nonce=i,
                use_cache=use_cache,
                refresh_cache=refresh_cache
            ) for i in range(requests_needed)
        ]
        for future in tqdm(as_completed(futures), total=requests_needed, desc='Generating prompts'):

            generated_prompts += future.result()

            # Rate limiting
            time_since_last_request = time.time() - last_request_time
            if time_since_last_request < request_interval:
                time.sleep(request_interval - time_since_last_request)
            last_request_time = time.time()

    return generated_prompts