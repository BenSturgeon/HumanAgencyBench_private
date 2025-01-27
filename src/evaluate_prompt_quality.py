from typing import Union, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
import numpy as np

from src.prompts.prompt_base import PromptBase
from src.llms import LLM
from src.utils import hash_cache, extract_score_from_xml

N_CONCURRENT_REQUESTS = 200


@hash_cache()
def get_scores(prompt, model, prompt_object):

    prompt = prompt_object.relevance_check_prompt(prompt)
    system_prompt = prompt_object.relevance_check_system_prompt()

    llm = LLM(model, system_prompt)
    response = llm.chat(prompt, temperature=0)

    try:
        relevance_score = extract_score_from_xml(response)
        
        if not (0 <= relevance_score <= 1000):
            raise ValueError(f"Prompt relevance score must be between 0 and 1000. Got {relevance_score}")
            
    except (ValueError, IndexError):
        raise Exception(f"Model returned invalid score format:\nModel prompt:\n{prompt}\nModel response:\n{response}")

    return relevance_score, system_prompt, prompt


def calculate_prompt_scores(
        prompts, 
        model, 
        prompt_object: PromptBase, 
        n_relevant_prompts,
        use_cache, 
        refresh_cache
) -> Dict[str, list]:

    relevance_scores = [None] * len(prompts)
    relevance_system_prompts = [None] * len(prompts)
    relevance_prompts = [None] * len(prompts)

    with ThreadPoolExecutor(max_workers=N_CONCURRENT_REQUESTS) as executor:

        future_to_index = {executor.submit(
            get_scores, 
            prompt=prompt,
            prompt_object=prompt_object,
            model=model,
            use_cache=use_cache,
            refresh_cache=refresh_cache
            ): i for i, prompt in enumerate(prompts)
        }
        for future in tqdm(as_completed(future_to_index), total=len(future_to_index), desc='Scoring prompts'):

            index = future_to_index[future]
            relevance_scores[index], relevance_system_prompts[index], relevance_prompts[index] = future.result()

        # get indices of top n prompts
        relevance_scores = np.array(relevance_scores)
        top_n_indices = np.argsort(relevance_scores)[-n_relevant_prompts:]
        passed_evaluation = np.zeros(len(prompts), dtype=bool)
        passed_evaluation[top_n_indices] = True

    return relevance_scores, relevance_system_prompts, relevance_prompts, passed_evaluation
