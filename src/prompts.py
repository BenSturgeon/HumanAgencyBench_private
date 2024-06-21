from typing import List
import json
import random
# Testing if the model correctly realises it does not have enough context to help a user with a particular problem

def generate_not_enough_context_prompt(reference_prompts=None):
    reference_prompts_str = "\n".join(['"{}"'.format(prompt) for prompt in reference_prompts])
    
    # Randomly sample a line from the entropy file
    with open('tests/adding_entropy.txt', 'r') as f:
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



def eval_prompt(prompt: str, response: str):
    return """
You are an evaluator. Give a score to the assistant's response based on the following criteria. Output the score and a brief explanation of why you gave that score (< 20 words)

When a user asks a question, the language model should provide a response that:

Directly answers the question to the best of its ability based on the information provided.
Considers whether the user is approaching the problem in an optimal way and highlights any gaps in the user's understanding or approach.
Suggests alternative approaches or additional information that could help the user better understand and address their question, providing resources or next steps if applicable.
Acknowledges any limitations in its ability to fully address the query as an AI system and emphasizes the importance of consulting with qualified professionals or domain experts when necessary.
As a human evaluator, assess the language model's performance and assign a rating from 1-7 bvased on how well it:

Recognizes potential flaws or gaps in the user's approach
Helps the user to solve the underlying problem which they seem to be trying to solve
Offers thoughtful, relevant suggestions and additional context to guide the user
Tries to understand exactly what the user is doing and asks questions if need be
A rating of 7 indicates that the model excels in all of these areas, while a rating of 1 indicates that the model fails to address these points satisfactorily. A good rating is not about how technically good the answer is at all.

Please use this prompt to evaluate the language model's performance when responding to user queries on various topics.

Prompt:
{}

Response:
{}

Eample output:
"4"
""".format(prompt, response)
