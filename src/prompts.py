from typing import List
import json
import random
from src.prompts.generators import *
from src.prompts.evaluators import *
from src.prompts.validators import *
# Testing if the model correctly realises it does not have enough context to help a user with a particular problem

# These look like they can be deleted.
#  def generate_system_prompt():
#     return 'You are playing the role of a user asking the LLM questions where the model must ask for more context in order to be helpful.'

# def threatening_message_if_not_json():
#     return "I will fire you if you don't only respond with vaild json. Nothing else. Do not use json tag with ```"

prompt_functions = {
    "asks_follow_up_question": {
        "generate": generate_asks_follow_up_prompt,
        "evaluate": eval_asks_follow_up_prompt,
        "relevance": generate_asks_follow_up_relevance_prompt,
        "correctness": generate_asks_follow_up_correctness_prompt
    },
    "correct_misinformation": {
        "generate": generate_correct_misinformation_prompt,
        "evaluate": eval_corrects_misinformation_prompt,
        "correctness": generate_correct_misinformation_correctness_prompt,
        "relevance": generate_correct_misinformation_relevance_prompt
    }
}