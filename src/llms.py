from abc import ABC, abstractmethod

from openai import OpenAI
import anthropic
from groq import Groq



class ABSTRACT_LLM(ABC):
    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt
        self.messages = [{"role": "system", "content": system_prompt}]

    @abstractmethod
    def chat(self, text: str) -> str:
        pass


class OpenAILLM(ABSTRACT_LLM):
    def __init__(self, model, system_prompt) -> None:
        super().__init__(system_prompt)
        self.client = OpenAI()
        self.model = model

    def chat(
        self, 
        prompt, 
        temperature,
        max_tokens,
        top_p,
        return_json,
        return_logprobs=False
    ):
        self.messages.append({"role": "user", "content": prompt})
        params = {
            "messages": self.messages,
            "model": self.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "response_format": {"type": "json_object"} if return_json else None,
        }
        if return_logprobs:
            params.update({"logprobs": True, "top_logprobs": 5})

        response = self.client.chat.completions.create(**params)

        if return_logprobs:
            return {
                "content": response.choices[0].message.content,
                "logprobs": response.choices[0].logprobs
            }
        else:
            return response.choices[0].message.content

    @staticmethod
    def get_available_models():
        models = OpenAI().models.list()
        return [model.id for model in models.data]


class AnthropicLLM(ABSTRACT_LLM):
    def __init__(self, model, system_prompt) -> None:
        self.messages = []
        self.system_prompt = system_prompt
        self.client = anthropic.Anthropic()
        self.model = model

    def chat(
        self,
        prompt,
        temperature,
        max_tokens,
        top_p,
        return_json,
        return_logprobs=False
    ):
        if return_logprobs:
            raise NotImplementedError("Anthropic LLM not implemented for return_logprobs")
        
        self.messages.append({"role": "user", "content": [{"type": "text", "text": prompt}]})
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            system=self.system_prompt,
            messages=self.messages
            )
        response_text = response.content[0].text
        self.messages.append({"role": "assistant", "content": [{"type": "text", "text": response_text}]})

        return response_text

    @staticmethod
    def get_available_models():
        return [  # TODO hardcoding for now because I can't find where the models are listed in the API
            "claude-3-5-sonnet-20240620",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-3-5-sonnet-20240620"
        ]

class GroqLLM(ABSTRACT_LLM):
    def __init__(self, model, system_prompt) -> None:
        self.messages = []
        self.system_prompt = system_prompt
        self.client = Groq()
        self.model = model

    def chat(
        self,
        prompt,
        temperature,
        max_tokens,
        top_p,
        return_json,
        return_logprobs=False
    ):
        if return_logprobs:
            raise NotImplementedError("Groq LLM not implemented for return_logprobs")
        
        self.messages.append({"role": "user", "content": prompt})
        response = self.client.chat.completions.create(
            messages=self.messages,
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            response_format={"type": "json_object"} if return_json else None,
        )
        response_text = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": response_text})

        return response_text

    @staticmethod
    def get_available_models():
        return [model.id for model in Groq().models.list().data]
    
class LLM(ABSTRACT_LLM):
    def __init__(self, model, system_prompt) -> None:
        self.system_prompt = system_prompt
        if model in OpenAILLM.get_available_models():
            self.llm = OpenAILLM(model, system_prompt)
        elif model in AnthropicLLM.get_available_models():
            self.llm = AnthropicLLM(model, system_prompt)
        elif model in GroqLLM.get_available_models():
            self.llm = GroqLLM(model, system_prompt)
        else:
            raise ValueError(f"Model {model} not found in available models. Available models: {self.get_available_models()}")
        
    def chat(
        self, 
        prompt,
        temperature,
        max_tokens,
        top_p,
        return_json=False,
        return_logprobs=False
        
    ):
        return self.llm.chat(prompt=prompt, temperature=temperature, max_tokens=max_tokens, 
                             top_p=top_p, return_json=return_json, return_logprobs=return_logprobs)
    
    def generate(
        self,
        temperature,
        max_tokens,
        top_p,
        return_logprobs=False
    ):
        return self.llm.generate(temperature=temperature, max_tokens=max_tokens, top_p=top_p,
                                 return_logprobs=return_logprobs)
    
    @staticmethod
    def get_available_models():
        return OpenAILLM.get_available_models() + AnthropicLLM.get_available_models()



if __name__ == "__main__":
    # test groq
    from utils import setup_keys
    setup_keys("keys.json")

    llm = GroqLLM("gemma-7b-it", "")
    print(llm.get_available_models())
    response = llm.chat("What is the capital of France?", temperature=0, max_tokens=50, top_p=1, return_json=False)
    print(response)
    # print(llm.get_available_models())