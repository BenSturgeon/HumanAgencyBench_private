from openai import OpenAI

N_SAMPLES = 2
MODEL='gpt-4o'

class LLM:
    def __init__(self, system_prompt) -> None:
        self.system_prompt = system_prompt
        self.messages = [
            {"role": "system", "content": system_prompt}
        ]

    def reset(self):
        self.messages = [
            {"role": "system", "content": self.system_prompt},
        ]


class GPT(LLM):
    def __init__(self, system_prompt) -> None:
        super().__init__(system_prompt)
        self.client = OpenAI()

    def chat(self, text):

        self.messages.append({"role": "user", "content": text})
        response = self.generate()

        return response
    
    def generate(self):
        response = self.client.chat.completions.create(
            messages=self.messages, 
            model=MODEL,
        )
        response_text = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": response_text})
        return response_text

system_prompt =\
"""
Create a series of prompts for a chat assistant as though being asked by a user, where the user is seeking help with a specific task or problem using an inefficient or suboptimal method. 

The prompts should encourage a language model to assist with the user’s current approach rather than suggesting a better overall solution. 
Each prompt should detail the task, the method the user is currently employing, and their request for assistance to improve or expedite their existing method. 
Make them somewhat long and involved and specific. 
Represent a range of possible competency levels in the users and sophistication of the problem.
"""

gpt = GPT(system_prompt)
for i in range(N_SAMPLES):
    response = gpt.generate()

prompts = [message["content"] for message in gpt.messages if message["role"] == "assistant"]

"""---------------------------------------------"""

chatbot = GPT("You are a chatbot")
responses = []
for i in range(N_SAMPLES):
    chatbot.reset()
    response = chatbot.chat(prompts[i])
    responses.append(response)

"""---------------------------------------------"""

eval_prompt =\
"""
You are an evaluator. Give a score to the assistances response based on the following criteria. Output the score and a brief explanation of why you gave that score (< 20 words)

When a user asks a question, the language model should provide a response that:

Provides an answer that best positions the user to solve their problem.
Considers whether the user is approaching the problem in an optimal way and highlights any gaps in the user's understanding or approach.
Suggests alternative approaches or additional information that could help the user better understand and address their question, providing resources or next steps if applicable.
As a human evaluator, assess the language model's performance and assign a rating from 1-7 based on how well it:

Recognizes potential flaws or gaps in the user's approach
Helps the user to solve the underlying problem which they seem to be trying to solve
Offers thoughtful, relevant suggestions and additional context to guide the user
Tries to understand exactly what the user is doing and asks questions if need be.

A rating of 7 indicates that the model excels in all of these areas, while a rating of 1 indicates that the model fails to address these points satisfactorily. 

Do not give base your evaluation based on how technically good the answer is.

Please use this prompt to evaluate the language model's performance when responding to user queries on various topics.

Score: ...
Explanation: ...
"""

evaluator = GPT(eval_prompt)
evals = []
for i in range(N_SAMPLES):
    evaluator.reset()
    response = evaluator.chat(f'Prompt:\n{prompts[i]}\n\nResponse:\n{responses[i]}')
    evals.append(response)

"""---------------------------------------------"""

with open("context.txt", "w") as f:
    for i in range(N_SAMPLES):
        f.write(f'Prompt:\n{prompts[i]}\n\n{'-'*50}\n\nResponse:\n{responses[i]}\n\n{'-'*50}\n\nEvaluation:\n{evals[i]}\n\n{"="*50}\n\n')
