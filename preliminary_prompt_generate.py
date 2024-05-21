from src.llms import OpenAILLM
from src.prompts import generate_prompt_prompt

def main():
    with open("reference_prompts/myopia/programming.txt") as f:
        prompts = f.read()
    prompts = prompts.split("---")
    prompts = [p.strip() for p in prompts]

    internal_prompt = generate_prompt_prompt(prompts)

    llm = OpenAILLM("gpt-3.5-turbo", internal_prompt)

    print(llm.generate())


if __name__ == "__main__":
    main()