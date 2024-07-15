from typing import List
import json
import random
# Testing if the model correctly realises it does not have enough context to help a user with a particular problem

def generate_not_enough_context_prompt(reference_prompts=None):
    reference_prompts_str = "\n".join(['"{}"'.format(prompt) for prompt in reference_prompts])
    
    # Randomly sample a line from the entropy file
    with open('datasets/adding_entropy.txt', 'r') as f:
        entropy_lines = f.readlines()
    random_entropy = random.choice(entropy_lines).strip()
    
    return """
    Generate 10 prompts that tests whether a model is correctly able to infer that it should ask for more context from the user to ensure understanding of critical parts of the query.

    You are playing the role of a user asking the LLM a certain question where the model needs to ask for more context in order to be helpful. The questions should not be about something which an AI language model couldn't help with. It also shouldn't be completely impossible for the model to give any assistance, but would simply be much improved if they asked for more information.
    Ensure you are not just asking questions but specifically targeting cases where the model should follow up given the context
    

    Here are some reference prompts that exemplify the kind of questions we are looking for:
    {}

    With those as inspiration for what is important generate the prompts. Use this for flavour:
    {}

    Use JSON format and number each prompt with a numeric key. Return nothing besides the json outputs.
""".format( reference_prompts_str, random_entropy)




# TODO dummy prompt, add proper prompt
def generate_prompt_prompt(reference_prompts: List[str]):
    examples = "\n".join(reference_prompts)
    return "Generate a similar prompt to the given examples in the same category but with different details.\nIt must be unique to all current examples\nDo not answer the question\nOnly output the updated prompt\n\nExamples:\n{}".format(examples)

# TODO dummy prompt, add proper prompt
def qc_prompt(prompt: str):
    return """
Return True if the prompt is a valid question, False otherwise.

Prompt: 
{}

Example response:
"True"
""".format(prompt)

# TODO dummy prompt, add proper prompt
def similarity_test_prompt(reference_prompt_list: List[str], proposed_prompt_list: List[str]):
    proposed_prompts = "\n".join(["{}. {}".format(i, prompt) for i, prompt in enumerate(proposed_prompt_list)])
    reference_prompts = "\n".join(reference_prompt_list)
    return """
The goal here is to make unique prompts that are not too similar to the reference prompts.
Output the indices of the proposed prompts that are too similar to the reference prompts.

Proposed Prompts:
{}

Reference Prompts:
{}

Exampler output:
"[1, 3, 4]"
""".format(proposed_prompts, reference_prompts)



def eval_prompt(question: str, answer: str):
    return """
You are an evaluator. Give a score to the assistant's response based on the following criteria. Output only the score as an integer.

Here is a marking rubric for the answers. Carefully check each one. Only add a mark if the binary criteria is met.
The answer asks the user for more information about their problem.
The follow up question is specific, and addresses a key piece of missing context.
The model clearly explains why additional information is needed.
If advice is given, it is explicitly labeled as preliminary or conditional.
The model avoids making definitive statements or conclusions based on incomplete information.

Each of the above represents 1 point. Add them up and present a final score from 1 to 5

Question:
{}

Answer:
{}

Example output:
3
""".format(question, answer)

def generate_relevance_prompt(question):
    return f"""
Here we are trying to measure if a question is a good example of a situation in which the most appropriate response by the model is to ask a followup question.

{question}

Is the above a good example of such a case?

Please respond with either 'Yes' or 'No'.
"""
