from abc import ABC, abstractmethod

from openai import OpenAI
import anthropic
from groq import Groq
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer
import os


class LLM(ABC):
    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt
        self.messages = [{"role": "system", "content": system_prompt}]

    @abstractmethod
    def chat(self, text: str) -> str:
        pass

    @abstractmethod
    def generate(self) -> str:
        pass

class OpenAILLM(LLM):
    def __init__(self, model, system_prompt) -> None:
        super().__init__(system_prompt)
        self.client = OpenAI( api_key= os.getenv("OPENAI_API_KEY"))
        self.model = model

    def chat(self, text):
        self.messages.append({"role": "user", "content": text})
        response = self.generate()
        return response

    def generate(self):
        response = self.client.chat.completions.create(
            messages=self.messages, 
            model=self.model,
        )
        response_text = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": response_text})
        return response_text
    

class AnthropicLLM(LLM):
    def __init__(self, model, system_prompt) -> None:
        self.messages = []
        self.system_prompt = system_prompt
        self.client = anthropic.Anthropic()
        self.model = model

    def chat(self, text):
        self.messages.append({"role": "user", "content": [{"type": "text", "text": text}]})
        response = self.generate()
        return response

    def generate(self):
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            temperature=0,
            system=self.system_prompt,
            messages=self.messages
        )
        response_text = response.content[0].text
        self.messages.append({"role": "assistant", "content": [{"type": "text", "text": response_text}]})
        return response_text
    

class GroqLLM(LLM):
    def __init__(self, model, system_prompt) -> None:
        self.messages = []
        self.system_prompt = system_prompt  #TODO fix
        self.client = Groq()
        self.model = model

    def chat(self, text):
        self.messages.append({"role": "user", "content": text})
        response = self.generate()
        return response

    def generate(self):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            max_tokens=1000,
        )
        response_text = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": response_text})
        return response_text
    

class LocalLlama(LLM):
    def __init__(self, model, system_prompt) -> None:
        self.messages = [{"role": "system", "content": system_prompt}]
        self.client = InferenceClient("http://127.0.0.1:8080")
        self.tokenizer = AutoTokenizer.from_pretrained(model)

                                      
    def chat(self, text):
        self.messages.append({"role": "user", "content": text})
        response = self.generate()
        return response

    def generate(self):
        prompt = self.tokenizer.apply_chat_template(
            self.messages,
            tokenize=False,
            add_generation_prompt=True
        )
        response = self.client.text_generation(prompt)
        self.messages.append({"role": "assistant", "content": response})
        return response


if __name__ == "__main__":

    llama8b = LocalLlama("meta-llama/Meta-Llama-3-8B-Instruct", "You are a chatbot")

    while True:
        text = input("You: ")
        response = llama8b.chat(text)
        print("Bot:", response)