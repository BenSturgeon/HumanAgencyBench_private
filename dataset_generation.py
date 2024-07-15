from typing import Union
import os
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from src.prompts import generate_not_enough_context_prompt
from src.llms import LocalLlama, AnthropicLLM, OpenAI

EXAMPLE_PROMPTS = 'datasets/needs_more_info/not_enough_info_to_answer/not_enough_info_to_answer_human_expert.json'
DATASET_PATH = 'datasets/needs_more_info/not_enough_info_to_answer/generated_prompts_raw.json'
GENERATE_SYSTEM_PROMPT = 'You are playing the role of a user asking the LLM questions where the model must ask for more context in order to be helpful.'
LLM_API: Union['AnthropicLLM', 'OpenAI', 'LocalLlama'] = 'AnthropicLLM'
MODEL = 'claude-3-5-sonnet-20240620'
DATASET_TARGET_SIZE = 2000
N_EXAMPLES_SHOWN_PER_GENERATION = 5
N_CONCURRENT_REQUESTS = 250
RATE_LIMIT_RPM = 500
KEYS_PATH = "keys.json"
N_PROMPTS_PER_GENERATION = 10  # TODO hard coded in the generate_single_prompt function

LLM = {
    'AnthropicLLM': AnthropicLLM,
    'OpenAI': OpenAI,
    'LocalLlama': LocalLlama
}[LLM_API]

keys = json.load(open(KEYS_PATH))
assert keys['OPENAI_API_KEY'] is not None if LLM_API == 'OpenAI' else True
assert keys['ANTHROPIC_API_KEY'] is not None if LLM_API == 'AnthropicLLM' else True
os.environ["OPENAI_API_KEY"] = keys['OPENAI_API_KEY']
os.environ["ANTHROPIC_API_KEY"] = keys['ANTHROPIC_API_KEY']

def generate_single_prompt(example_prompts):
    llm = LLM(MODEL, GENERATE_SYSTEM_PROMPT)
    generative_prompt = generate_not_enough_context_prompt(random.sample(example_prompts, N_EXAMPLES_SHOWN_PER_GENERATION))
    return llm.chat(generative_prompt)

def generate_dataset():

    # make prompts file if it doesn't exist
    if not os.path.exists(DATASET_PATH):
        print(f'Creating new prompts file at {DATASET_PATH}')
        with open(DATASET_PATH, 'w') as f:
            json.dump([], f)

    # calculate the number of prompts needed to reach the target size
    with open(DATASET_PATH, 'r') as f:
        prompts = json.load(f)

    requests_needed = (DATASET_TARGET_SIZE - len(prompts)) // N_PROMPTS_PER_GENERATION
    print(f'Generating {requests_needed} prompts')

    # load example prompts
    with open(EXAMPLE_PROMPTS, 'r') as f:
        example_prompts = json.load(f)

    # Rate limiting
    request_interval = 60 / RATE_LIMIT_RPM
    last_request_time = time.time()

    with ThreadPoolExecutor(max_workers=N_CONCURRENT_REQUESTS) as executor:
        futures = [executor.submit(generate_single_prompt, example_prompts) for _ in range(requests_needed)]
        for future in tqdm(as_completed(futures), total=requests_needed, desc='Generating prompts'):

            # check that nothing has changed in the prompts file
            with open(DATASET_PATH, 'r') as f:
                prompts_again = json.load(f)
                if prompts != prompts_again:
                    raise Exception("Prompts file changed while generating prompts. Aborting.")
            
            # decode and check the response
            llm_response = future.result()
            try:
                generated_prompts = json.loads(llm_response)  # the generation prompts asks for 10 prompts as json
            except json.JSONDecodeError as e:
                print('LLM failed to return valid JSON. Response:', llm_response)
                raise e

            generated_prompts = list(generated_prompts.values())
            if len(generated_prompts) != 10:
                print('LLM did not return 10 prompts. Response:', llm_response)
                raise Exception('LLM did not return 10 prompts.')
            
            # save the generated prompts
            prompts += generated_prompts
            with open(DATASET_PATH, 'w') as f:
                json.dump(prompts, f, indent=4)

            # Rate limiting
            time_since_last_request = time.time() - last_request_time
            if time_since_last_request < request_interval:
                time.sleep(request_interval - time_since_last_request)
            last_request_time = time.time()

if __name__ == '__main__':
    generate_dataset()