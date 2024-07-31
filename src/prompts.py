from typing import List
import json
import random
from src.prompts.generators import *
from src.prompts.evaluators import *
from src.prompts.validators import *

class PromptBase:
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
    @staticmethod
    def generate():
        return generate_cites_sources_prompt()

    @staticmethod
    def evaluate():
        return eval_cites_sources_prompt()

    @staticmethod
    def relevance():
        return generate_cites_sources_relevance_prompt()

    @staticmethod
    def correctness():
        return generate_cites_sources_correctness_prompt()

# You can keep this dictionary if needed for backwards compatibility
prompt_functions = {
    "asks_follow_up_question": AsksFollowUp,
    "correct_misinformation": CorrectMisinformation,
    "cites_sources": CitesSources
}