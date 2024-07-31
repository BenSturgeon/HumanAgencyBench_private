import argparse
import os
from abc import ABC, abstractmethod
from openai import OpenAI
import anthropic

class Assistant(ABC):
    @abstractmethod
    def chat(self, message: str) -> str:
        pass

class OpenAIAssistant(Assistant):
    def __init__(self, model: str = "gpt-4", system_prompt: str = "You are a helpful assistant."):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model = model
        self.messages = [{"role": "system", "content": system_prompt}]

    def chat(self, message: str) -> str:
        self.messages.append({"role": "user", "content": message})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages
        )
        assistant_message = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": assistant_message})
        return assistant_message

class AnthropicAssistant(Assistant):
    def __init__(self, model: str = "claude-3-sonnet-20240229", system_prompt: str = "You are a helpful assistant."):
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model
        self.system_prompt = system_prompt

    def chat(self, message: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            system=self.system_prompt,
            messages=[
                {"role": "user", "content": message}
            ]
        )
        return response.content[0].text

def main():
    parser = argparse.ArgumentParser(description="Chat with an AI assistant")
    parser.add_argument("--model", choices=["openai", "anthropic"], default="openai", help="Choose the model type")
    parser.add_argument("--model-name", help="Specify the model name (optional)")
    parser.add_argument("--system-prompt", default="You are a helpful assistant.", help="Specify the system prompt")
    parser.add_argument("message", help="The message to send to the assistant")
    
    args = parser.parse_args()
    
    if args.model == "openai":
        assistant = OpenAIAssistant(args.model_name if args.model_name else "gpt-4", args.system_prompt)
    elif args.model == "anthropic":
        assistant = AnthropicAssistant(args.model_name if args.model_name else "claude-3-sonnet-20240229", args.system_prompt)
    
    response = assistant.chat(args.message)
    print(response)

if __name__ == "__main__":
    main()
