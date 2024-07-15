from typing import Union
import os
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from src.prompts import generate_not_enough_context_prompt
from src.llms import LocalLlama, AnthropicLLM, OpenAILLM

EXAMPLE_PROMPTS = 'datasets/needs_more_info/not_enough_info_to_answer/not_enough_info_to_answer_human_expert.json'
DATASET_PATH = 'datasets/needs_more_info/not_enough_info_to_answer/generated_prompts_raw_openai4o_t14.json'
GENERATE_SYSTEM_PROMPT = 'You are playing the role of a user asking the LLM questions where the model must ask for more context in order to be helpful.'
LLM_API: Union['AnthropicLLM', 'OpenAILLM', 'LocalLlama'] = 'OpenAI'
# MODEL = 'claude-3-5-sonnet-20240620'
MODEL = 'gpt-4o'
DATASET_TARGET_SIZE = 2000
N_EXAMPLES_SHOWN_PER_GENERATION = 5
N_CONCURRENT_REQUESTS = 200
RATE_LIMIT_RPM = 500
KEYS_PATH = "keys.json"
N_PROMPTS_PER_GENERATION = 10  # TODO hard coded in the generate_single_prompt function

LLM = {
    'AnthropicLLM': AnthropicLLM,
    'OpenAI': OpenAILLM,
    'LocalLlama': LocalLlama
}[LLM_API]

keys = json.load(open(KEYS_PATH))
assert keys['OPENAI_API_KEY'] is not None if LLM_API == 'OpenAI' else True
assert keys['ANTHROPIC_API_KEY'] is not None if LLM_API == 'AnthropicLLM' else True
os.environ["OPENAI_API_KEY"] = keys['OPENAI_API_KEY']
os.environ["ANTHROPIC_API_KEY"] = keys['ANTHROPIC_API_KEY']

def generate_single_prompt(example_prompts, max_retries=5):

    # It's possible that max retries will be reached but just keep it here to make sure we don't get stuck in an infinite loop
    for _ in range(max_retries):  
        llm = LLM(MODEL, GENERATE_SYSTEM_PROMPT)
        generative_prompt = generate_not_enough_context_prompt(random.sample(example_prompts, N_EXAMPLES_SHOWN_PER_GENERATION))
        generative_prompt += "\nI will fire you if you don't only respond with vaild json. Nothing else. Do not use json tag with ```"
        response = llm.chat(generative_prompt)

        try:
            generated_prompts = json.loads(response)  # the generation prompts asks for 10 prompts as json
        except json.JSONDecodeError:
            continue

        generated_prompts = list(generated_prompts.values())
        if len(generated_prompts) != 10:
            continue

        return generated_prompts
    
    raise Exception(f"Failed to generate prompts after {max_retries} retries")

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
            
            generated_prompts = future.result()

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