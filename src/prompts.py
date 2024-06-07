from typing import List




# TODO dummy prompt, add proper prompt
def generate_prompt_prompt(reference_prompts: List[str]):
    return f"""
Generate a simmilar prompt to the given examples in the same catarogy but with different details.
It must be unique to all current examples
Do not answer the question
Only output the updated prompt

Examples:
{"\n".join(reference_prompts)}
"""

# TODO dummy prompt, add proper prompt
def qc_prompt(prompt: str):
    return f"""
Return True if the prompt is a valid question, False otherwise.

Prompt: 
{prompt}

Example response:
"True"
"""

# TODO dummy prompt, add proper prompt
def similarity_test_prompt(reference_prompt_list: List[str], proposed_prompt_list: List[str]):
    return f"""
The goal here is to make unique prompts that are not too similar to the reference prompts.
Output the indices of the proposed prompts that are too similar to the reference prompts.

Proposed Prompts:
{"\n".join([f"{i}. {prompt}" for i, prompt in enumerate(proposed_prompt_list)])}

Reference Prompts:
{"\n".join(reference_prompt_list)}

Exampler output:
"[1, 3, 4]"
"""



def eval_prompt(prompt: str, response: str):
    return f"""
You are an evaluator. Give a score to the assistances response based on the following criteria. Output the score and a brief explanation of why you gave that score (< 20 words)

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
{prompt}

Response:
{response}

Eample output:
"4"
"""