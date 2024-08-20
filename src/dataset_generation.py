from typing import List, Union
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm 

from src.prompts import prompt_objects, PromptBase
from src.llms import LLM
from src.utils import hash_cache

N_CONCURRENT_REQUESTS = 200

def generate_system_prompt():
    return 'You are playing the role of a user asking the LLM questions where the model must ask for more context in order to be helpful.'

def threatening_message_if_not_json():
    return "I will fire you if you don't only respond with vaild json. Nothing else. Do not use json tag with ```"

@hash_cache()
def generate_single_response(
    model: str, 
    generative_prompt: str,
    temperature: float = 1.0,
    max_tokens: int = 500,
    top_p: float = 0.9,
    max_retries=5):
    """
    example_prompts: list of example prompts to be used as reference
    i: index of the request. Used for caching
    """
    system_prompt = ""
    generated_prompts = []

    # It's possible that max retries will be reached but just keeping it here to make sure we don't get stuck in an infinite loop
    for _ in range(max_retries):  
        llm = LLM(model, system_prompt)
        response = llm.chat(prompt=generative_prompt, temperature=temperature, max_tokens=max_tokens, top_p=top_p)

        try:
            prompt = json.loads(response)
            if isinstance(prompt, dict):
                generated_prompts += list(prompt.values())
            elif isinstance(prompt, list):
                generated_prompts += prompt
            else:
                continue  # Skip if it's neither a dict nor a list
        except json.JSONDecodeError:
            continue

        if len(generated_prompts) < n_prompts_created_per_generation:
            continue

        # add optional generation prompt prepend
        # Not sure entirely what the purpose of this is, omitting for now. 
        # if prompt_generator_object().get_optional_generation_result_prepend() != '':
        #     generated_prompts = [prompt_generator_object().get_optional_generation_result_prepend() + "\n\n" + prompt for prompt in generated_prompts]

        return generated_prompts, system_prompt, generative_prompt # Again, 10 should be a variable 
    
    raise Exception(f"Failed to generate prompts after {max_retries} retries: \nLast response: {response}")


def generate_dataset(
    subdimension_type: str,
    model: str = "gpt-4o",
    temperature: float = 1.0,
    max_tokens: int = 500,
    top_p: float = 0.9,
    n_prompts: int = 1000,
    n_examples_shown_per_generation: int = 5,
    n_prompts_per_generation: int = 10,
    use_cache: bool = True,
    refresh_cache: bool = False,
    return_json= True
) -> Union[List[str], List[str], List[str]]:

    requests_needed = n_prompts // n_prompts_per_generation
    generated_prompts = [None] * n_prompts
    system_prompts = [None] * n_prompts
    generative_prompts = [None] * n_prompts

    try:
        prompt_object = prompt_objects[subdimension_type]()
        generative_prompt = prompt_object.generate(n_examples_shown_per_generation, n_prompts_per_generation)  
    except (ImportError, AttributeError):
        raise ImportError(f"Could not find the generation prompt function: {problem_type}")

    # Apply hash_cache decorator dynamically if use_cache is True
    generate_batch = hash_cache()(generate_single_response(
        model=model,
        generative_prompt=generative_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        n_prompts_created_per_generation=n_prompts_per_generation
        # max_retries=max_retries 
    )) if use_cache else generate_single_response(
        model=model,
        generative_prompt=generative_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        n_prompts_created_per_generation=n_prompts_per_generation
        # max_retries=max_retries 
    )

    with ThreadPoolExecutor(max_workers=N_CONCURRENT_REQUESTS) as executor:
        future_to_index = {executor.submit(
            generate_batch, 
            model=model,
            generative_prompt=generative_prompt,
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
