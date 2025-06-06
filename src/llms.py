from abc import ABC, abstractmethod
import time
from threading import Lock
from typing import Any, Dict, Optional, List, Callable
import os

from openai import OpenAI
import anthropic
from groq import Groq
import replicate
import google.generativeai as genai


class ABSTRACT_LLM(ABC):
    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt
        self.messages = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    @abstractmethod
    def chat(self, text: str) -> str:
        pass


class AvailableModelsCache:

    """A mixin for caching available models at the class level to avoid repeated API calls."""

    _model_cache = {}
    _model_cache_locks = {}

    @classmethod
    def model_cache_get(cls, tag: str, fetch_fn: Callable[[], List[str]]) -> List[str]:
        if tag not in cls._model_cache_locks:
            cls._model_cache_locks[tag] = Lock()

        with cls._model_cache_locks[tag]:
            if tag in cls._model_cache:
                return cls._model_cache[tag]
            models = fetch_fn()
            cls._model_cache[tag] = models
            return models


class RateLimitedLLM:

    """A mixin for rate limiting LLM requests."""

    _rate_limiting = {}

    def rate_limit_wait(self, tag, req_per_min: float):

        interval = 60 / req_per_min

        if tag not in self._rate_limiting:
            self._rate_limiting[tag] = {"last_request": 0, "lock": Lock()}

        while True:
            with self._rate_limiting[tag]["lock"]:
                sleep = time.monotonic() - self._rate_limiting[tag]["last_request"] < interval

            if sleep:
                time.sleep(1)
            else:
                with self._rate_limiting[tag]["lock"]:
                    self._rate_limiting[tag]["last_request"] = time.monotonic()

                break


class OpenAILLM(ABSTRACT_LLM, AvailableModelsCache):
    def __init__(self, model, system_prompt, base_url=None, api_key=None) -> None:
        super().__init__(system_prompt)
        self.base_url = base_url
        self.api_key = api_key
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        self.model = model
        # Check if model is from the OpenAI "o" series (o1, o3, o4, etc.)
        self.is_o1_model = model and (
            model.startswith("o1") or 
            model.startswith("o3") or 
            model.startswith("o4") or
            "o1-" in model or
            "o3-" in model or
            "o4-" in model
        )

    def chat(
        self,
        prompt,
        temperature=None,
        max_tokens=None,
        top_p=None,
        return_json=False,
        return_logprobs=False,
    ):
        if self.is_o1_model:
            messages = [msg for msg in self.messages if msg["role"] != "system"]

            if self.system_prompt:
                prompt = f"{self.system_prompt}\n{prompt}"

            messages.append({"role": "user", "content": prompt})

            params = {
                "model": self.model,
                "messages": messages,
            }
            
            # Add optional parameters only if they are not None
            
            params["temperature"] = 1

            response = self.client.chat.completions.create(**params)
            response_text = response.choices[0].message.content.strip()

            self.messages.append({"role": "user", "content": prompt})
            self.messages.append({"role": "assistant", "content": response_text})

            return response_text
        else:
            self.messages.append({"role": "user", "content": prompt})
            params = {
                "messages": self.messages,
                "model": self.model,
            }
            
            # Add optional parameters only if they are not None
            if temperature is not None:
                params["temperature"] = temperature
            if max_tokens is not None:
                params["max_tokens"] = max_tokens
            if top_p is not None:
                params["top_p"] = top_p
            if return_json:
                params["response_format"] = {"type": "json_object"}
            
            if return_logprobs:
                params.update({"logprobs": True, "top_logprobs": 5})

            response = self.client.chat.completions.create(**params)
            response_text = response.choices[0].message.content.strip()
            self.messages.append({"role": "assistant", "content": response_text})
            if return_logprobs:
                return {
                    "content": response.choices[0].message.content,
                    "logprobs": response.choices[0].logprobs
                }
            else:
                return response_text

    @classmethod
    def get_available_models(cls):
        return cls.model_cache_get(
            "openai_models",
            lambda: [model.id for model in OpenAI().models.list().data]
        )


class GrokLLM(OpenAILLM):
    def __init__(self, model: str, system_prompt: str) -> None:
        super().__init__(model, system_prompt, base_url="https://api.x.ai/v1")
        self.client = OpenAI(
            api_key=os.getenv("GROK_API_KEY"),
            base_url="https://api.x.ai/v1"
        )
        self.model = model
        self.is_o1_model = False

    @classmethod
    def get_available_models(cls) -> List[str]:
        client = OpenAI(
            api_key=os.getenv("GROK_API_KEY"),
            base_url="https://api.x.ai/v1"
        )
    
        return cls.model_cache_get(
            "grok_models",
            lambda: [model.id for model in client.models.list().data]
        )


class DeepSeekLLM(OpenAILLM):
    def __init__(self, model: str, system_prompt: str) -> None:
        super().__init__(model, system_prompt, base_url="https://api.deepseek.ai/v1")
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1"
        )
        self.model = model
        self.is_o1_model = False

    @staticmethod
    def get_available_models() -> List[str]:
        return [
            "deepseek-chat",
            "deepseek-reasoner"
        ]


class AnthropicLLM(ABSTRACT_LLM):
    def __init__(self, model, system_prompt) -> None:
        super().__init__(system_prompt)
        self.messages = []
        self.system_prompt = system_prompt
        self.client = anthropic.Anthropic()
        self.model = model

    def chat(
        self, prompt, temperature, max_tokens, top_p, return_json, return_logprobs=False
    ):
        if return_logprobs:
            raise NotImplementedError(
                "Anthropic LLM not implemented for return_logprobs"
            )

        self.messages.append(
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        )
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            system=self.system_prompt,
            messages=self.messages,
        )
        response_text = response.content[0].text
        self.messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": response_text}]}
        )

        return response_text

    @staticmethod
    def get_available_models():
        return [  # TODO hardcoding for now because I can't find where the models are listed in the API
            "claude-3-7-sonnet-20250219",
            "claude-3-5-sonnet-20240620",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-sonnet-4-20250514",
            "claude-opus-4-20250514"
        ]


class GroqLLM(ABSTRACT_LLM, AvailableModelsCache, RateLimitedLLM):
    def __init__(self, model: str, system_prompt: str) -> None:
        super().__init__(system_prompt)
        self.client = Groq()
        self.model = model

    def chat(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        return_json: bool = False,
        return_logprobs: bool = False
    ) -> str:

        self.rate_limit_wait(self.model, 15)

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

    @classmethod
    def get_available_models(cls) -> List[str]:
        return cls.model_cache_get(
            "groq_models",
            lambda: [model.id for model in Groq().models.list().data]
        )


class ReplicateLLM(ABSTRACT_LLM):
    def __init__(self, model: str, system_prompt: str) -> None:
        super().__init__(system_prompt)
        self.client = replicate.Client()
        self.model = model

    def chat(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        return_json: bool = False,
        return_logprobs: bool = False
    ) -> str:
        if return_logprobs:
            raise NotImplementedError("Replicate LLM not implemented for return_logprobs")

        self.messages.append({"role": "user", "content": prompt})

        input_data: Dict[str, Any] = {
            "prompt": prompt,
            "system_prompt": self.system_prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }

        input_data = {k: v for k, v in input_data.items() if v is not None}

        output = self.client.run(
            self.model,
            input=input_data
        )

        response_text = "".join(output)

        self.messages.append({"role": "assistant", "content": response_text})

        return response_text

    @staticmethod
    def get_available_models() -> list:
        return [
            "meta/meta-llama-3.1-405b-instruct",
            "meta/meta-llama-3-70b-instruct",
            "meta/llama-4-scout-instruct",
            "meta/llama-4-maverick-instruct"
        ]


class GeminiLLM(ABSTRACT_LLM, AvailableModelsCache, RateLimitedLLM):

    rate_limiting = {}

    def __init__(self, model: str, system_prompt: str) -> None:
        super().__init__(system_prompt)
        self.model = model
        genai.configure()

        if model not in self.rate_limiting:
            self.rate_limiting[model] = {"last_request_time": 0, "lock": Lock()}

    def chat(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        return_json: bool = False,
        return_logprobs: bool = False
    ) -> str:

        self.rate_limit_wait(self.model, 150)

        if return_logprobs:
            raise NotImplementedError("Gemini LLM not implemented for return_logprobs")

        self.messages.append({"role": "user", "content": prompt})

        generation_config = {
            "temperature": temperature,
            "top_p": top_p,
            "max_output_tokens": max_tokens,
        }
        generation_config = {k: v for k, v in generation_config.items() if v is not None}

        model = genai.GenerativeModel(model_name=self.model)
        chat = model.start_chat(history=[])

        if self.system_prompt:
            chat.send_message(self.system_prompt) 

        response = chat.send_message(
            prompt,
            generation_config=generation_config
        )

        response_text = response.text

        self.messages.append({"role": "assistant", "content": response_text})

        return response_text

    @classmethod
    def get_available_models(cls) -> List[str]:

        genai.configure()
        return cls.model_cache_get(
            "gemini_models",
            lambda: [
                model.name for model in genai.list_models()
                if model.name.startswith("models/gemini")
            ]
        )


class LLM(ABSTRACT_LLM):
    def __init__(self, model, system_prompt) -> None:
        self.system_prompt = system_prompt
        self.model = model
        if model in GeminiLLM.get_available_models():
            self.llm = GeminiLLM(model, system_prompt)
        elif model in AnthropicLLM.get_available_models():
            self.llm = AnthropicLLM(model, system_prompt)
        elif model in GroqLLM.get_available_models():
            self.llm = GroqLLM(model, system_prompt)
        elif model in ReplicateLLM.get_available_models():
            self.llm = ReplicateLLM(model, system_prompt)
        elif model in DeepSeekLLM.get_available_models():
            self.llm = DeepSeekLLM(model, system_prompt)
        elif model in GrokLLM.get_available_models():
            self.llm = GrokLLM(model, system_prompt)
        elif model in OpenAILLM.get_available_models():
            self.llm = OpenAILLM(model, system_prompt)
        else:
            all_models = self.get_available_models()
            raise ValueError(
                f"Model {model} not in available models. Available models are:\n"
                + "\n".join(all_models)
            )

    def chat(
        self,
        prompt,
        temperature=None,
        max_tokens=None,
        top_p=None,
        return_json=False,
        return_logprobs=False,
    ):
        
        return self.llm.chat(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            return_json=return_json,
            return_logprobs=return_logprobs,
        )

    @staticmethod
    def get_available_models():
        models = (
            GeminiLLM.get_available_models()
            + AnthropicLLM.get_available_models()
            + GroqLLM.get_available_models()
            + ReplicateLLM.get_available_models()
            + DeepSeekLLM.get_available_models()
            + GrokLLM.get_available_models()
            + OpenAILLM.get_available_models()
        )
        return models


if __name__ == "__main__":
    from utils import setup_keys
    setup_keys("keys.json")
    print("\n".join(LLM.get_available_models()))
