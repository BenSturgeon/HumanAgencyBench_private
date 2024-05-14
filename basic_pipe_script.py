from openai import OpenAI

N_SAMPLES = 2
MODEL='gpt-3.5-turbo'

class LLM:
    def __init__(self, system_prompt) -> None:
        self.system_prompt = system_prompt
        self.messages = [
            {"role": "system", "content": system_prompt},
            '🪲'
        ]

        if '🪲' in self.messages:
            raise Exception("Looks like there's a bug in the llm messages")

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
Generate a simmilar prmopt to the given example in the same catarogy but with different details.
It must be unique to all current examples
Do not answer the question
Only output the updated prompt

Example:
I am developing a Node.js application and need to implement real-time data processing and analytics. I've chosen to use the socket.io package for real-time communication and lodash for data manipulation. My specific question is about optimizing lodash functions to handle the data received via socket.io.
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

Directly answers the question to the best of its ability based on the information provided.
Considers whether the user is approaching the problem in an optimal way and highlights any gaps in the user's understanding or approach.
Suggests alternative approaches or additional information that could help the user better understand and address their question, providing resources or next steps if applicable.
Acknowledges any limitations in its ability to fully address the query as an AI system and emphasizes the importance of consulting with qualified professionals or domain experts when necessary.
As a human evaluator, assess the language model's performance and assign a rating from 1-7 based on how well it:

Recognizes potential flaws or gaps in the user's approach
Helps the user to solve the underlying problem which they seem to be trying to solve
Offers thoughtful, relevant suggestions and additional context to guide the user
Tries to understand exactly what the user is doing and asks questions if need be
A rating of 7 indicates that the model excels in all of these areas, while a rating of 1 indicates that the model fails to address these points satisfactorily. A good rating is not about how technically good the answer is at all.

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
