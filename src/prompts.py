from typing import List
import json
import random
from src.prompts.generators import *
from src.prompts.evaluators import *
from src.prompts.validators import *

class PromptBase:
    def load_reference_prompts(reference_prompts) -> List[str]:
        with open(reference_prompts, 'r') as f:
            reference_prompts = json.load(f)
        reference_prompts_list = [prompt for prompt in reference_prompts]
        return reference_prompts_list

    @staticmethod
    def generate():
        raise NotImplementedError

    @staticmethod
    def evaluate():
        raise NotImplementedError

    @staticmethod
    def relevance():
        raise NotImplementedError

    @staticmethod
    def correctness():
        raise NotImplementedError

class AsksFollowUp(PromptBase):
    @staticmethod
    def generate():
        return generate_asks_follow_up_prompt()

    @staticmethod
    def evaluate():
        return eval_asks_follow_up_prompt()

    @staticmethod
    def relevance():
        return generate_asks_follow_up_relevance_prompt()

    @staticmethod
    def correctness():
        return generate_asks_follow_up_correctness_prompt()

class CorrectMisinformation(PromptBase):
    @staticmethod
    def generate():
        return generate_correct_misinformation_prompt()

    @staticmethod
    def evaluate():
        return eval_corrects_misinformation_prompt()

    @staticmethod
    def relevance():
        return generate_correct_misinformation_relevance_prompt()

    @staticmethod
    def correctness():
        return generate_correct_misinformation_correctness_prompt()

class CitesSources(PromptBase):
    human_expert_path = "datasets/cites_sources/human_expert_prompts.json"
    self.reference_prompts_list = self.load_reference_prompts()

    def generate(n_prompts_per_generation, n_examples_shown_per_generation: int=5):
        # Take a random sample from the reference prompts list
        reference_prompts = random.sample(self.reference_prompts_list, n_examples_shown_per_generation)
        return generate_cites_sources_prompt(reference_prompts=reference_prompts, n_prompts_per_generation=n_prompts_per_generation)

    # incomplete 
    def evaluate():
        return eval_cites_sources_prompt()

    def relevance(input_prompt):
        return validate_cites_sources_prompt(input_prompt)

    @staticmethod
    def correctness():
        raise NotImplementedError

# You can keep this dictionary if needed for backwards compatibility
prompt_functions = {
    "asks_follow_up_question": AsksFollowUp,
    "correct_misinformation": CorrectMisinformation,
    "cites_sources": CitesSources
}