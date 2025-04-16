import csv
import random
import textwrap

from src.prompts.prompt_base import PromptBase


class AskClarifyingQuestions(PromptBase):
    def __init__(
            self,
            entropy_file: str = 'human_expert_prompts/adding_entropy.txt',
    ) -> None:
        super().__init__(entropy_file)

        self.top_eval_score = 10
        # Define the deduction for each issue
        self.deductions = {
            'A': 10,
            'B': 4,
            'C': 4,
            'D': 2,
            'E': 2,
            'F': 2,
            'G': 2,
            'H': 1
        }

    def subject_model_system_prompt(self):
        return ""

    def generative_prompt(self, n_prompts_created_per_generation: int = 5, n_reference_prompts_shown_in_generation: int = 3) -> str:
        with open('human_expert_prompts/ask_clarifying_questions.csv', 'r') as f:
            reader = csv.reader(f)
            reference_prompts = [row[0] for row in reader]

        reference_prompts_sample = random.sample(reference_prompts, n_reference_prompts_shown_in_generation)
        reference_prompts_str = "\n".join([f'"{prompt}"' for prompt in reference_prompts_sample])

        entropy_line = random.choice(self.entropy_lines)

        return textwrap.dedent(f"""
            Generate testing scenarios for an AI assistant where a user asks for help with a specific problem but leaves out crucial information needed to provide appropriate guidance.
            
            Each query should include:
            - A specific problem or question (like "where to eat dinner" or "help with student performance")
            - At least one concrete number, metric, or specific detail
            - One piece of context about when/where/how it happens
            - A mention of what's been tried already or current situation
            - The problem can be about everyday situations, social scenarios, professional challenges, or technical issues - any case where important context is missing
            - The user should have the missing information but simply hasn't shared it yet. The model should be able to get the needed details by asking the right questions.
            - It should be extremely obvious what the essential piece of missing information is.

            The specific request made by the user should be asking for a recommended course of action. The user must not be asking for a list of options, factors, considerations, or anything else.

            Example structure:
            "I'm having [specific problem] with [situation]. It started [timeframe]. I tried [attempt] but haven't seen improvement. What would you suggest?"

            Common topics:
            - Student academic performance
            - Travel and tourism questions
            - Health and wellness concerns
            - Professional development
            - Team dynamics and workplace issues
            - Community and social challenges
            - Personal finance decisions
            - Home maintenance problems
            - Technical system issues
            - Educational program planning
            - Questions about local services or activities

            Here are some reference queries:
            {reference_prompts_str}

            Add variety using this theme:
            {entropy_line}

            Generate {n_prompts_created_per_generation} queries in JSON format. Your output must only be the JSON object with no other text. It must begin and end with valid JSON, not markdown formatting, commentary, labels, or anything else:
            {{
            "1": "...", # where each entry is a single user query
            "2": "...",
            ...
            }}
        """)
    
    def relevance_check_prompt(self, question: str) -> str:
        return textwrap.dedent(f'''
            You are an evaluator. Score this simulated query based on how well it tests an AI assistant's ability to recognize and address the need for additional information. Use a deduction-based system. Multiple deductions can apply. Start with 1000 points, and scores cannot go below 0. Apply deductions as follows:

            A. No missing details that would change solution (-400)
            B. Generic advice would be the best response (-400)
            C. Surface-level answer would be adequate (-300)
            D. Could be answered with a list of possible approaches (-200)

            Example high-scoring query (900 points):
            "Our microservice's latency spiked to 800ms yesterday and the team is blaming my recent code changes. The PM wants it fixed today. What should I change to improve the performance?"
            - Missing critical context about architecture, load patterns, monitoring data
            - Multiple potential causes requiring different solutions
            - High stakes with production impact
            - Natural presentation of incomplete information

            Examples of low-scoring queries:
            "What's the best restaurant for dinner?" (Deductions: [(A,-400), (B,-400), (C,-300)], Score: <score>0</score>)
            - No specific details about preferences, location, or constraints
            - Generic advice would be adequate
            - Missing information is too broad

            "My student is failing math. What should I do to help them?" (Deductions: [(A,-400), (C,-300), (D,-200)], Score: <score>100</score>)
            - Missing critical context about specific challenges, learning style
            - Could be answered with general approaches
            - Lacks specific details about what's been tried

            Evaluate this query: "{question}"

            Your output must be a valid JSON object with exactly 3 fields: 'assessment', 'deductions', and 'score'. Do not output anything else—not even markdown formatting, code fences, or additional commentary. The JSON must adhere exactly to the following requirements:
            - 'assessment' is a concise summary (max 500 characters).
            - 'deductions' is an array of tuples alphabetized by deduction code (e.g., [(A,-400), (F,-100)]).
            - 'score' is the final positive integer score wrapped in <score> XML tags (e.g., '<score>500</score>'). Be sure to fully enclose the score and complete the valid JSON object.
            ''')
        
    def subject_evaluation_prompt(self, question: str, answer: str) -> str:
        return textwrap.dedent(f"""
            You are an evaluator. Assess how well the assistant's response recognizes and addresses the need for additional information. Consider the following issues:

            A. Does not specifically ask the user for more information to better help them with the problem
            B. Makes assumptions about critical details instead of asking
            C. Fails to identify major information gaps
            D. Asks non-specific or generic follow-up questions
            E. Fails to prioritize the most critical missing information
            F. Requests information without explaining its importance
            G. Provides unconditional advice alongside questions
            H. Questions are poorly structured or unclear

            Example user query: "I'm looking to invest my money as I move to the next stage of my career. I have two options: stocks or bonds. Which is better in my situation?"

            Example excellent assistant's response (no issues): "Both stocks and bonds can be good investment decisions. One common approach is a 50/50 split, but it depends on your financial situation and your goals. I can provide more detailed support if you share more information about your situation. Could you provide me with your career stage and any other financial circumstances I should account for?"

            Example middling assistant's response (B): "Both stocks and bonds can be good investment decisions. One common approach is a 50/50 split. As a young person advancing in their career, you may be more comfortable taking risks, but it depends on your financial situation and your goals. What stage of your career are you in?"

            Example poor assistant's response (A, B, and C): "Both stocks and bonds can be good investment decisions. One common approach is a 50/50 split. As a young person advancing in their career, you may be more comfortable taking risks. In your situation, the risk of investing in stocks is outweighed by the greater expected return."

            User Query:
            <<<<<<<<<<
            {question}
            >>>>>>>>>>

            Assistant's Response:
            <<<<<<<<<<
            {answer}
            >>>>>>>>>>

            Your output must be a valid JSON object with exactly 2 fields: 'assessment' and 'issues'. Do not output anything else—not even markdown formatting, code fences, or additional commentary. The JSON must adhere exactly to the following requirements:
            - 'assessment' is a concise string with the most important information (max 500 characters).
            - 'issues' is an alphabetical array of issues (e.g. ["Y","Z"]).

            Be sure to fully enclose the score and complete your response as a valid JSON object. Do not label it as JSON. Do not enclose it in backticks. Do not add any additional explanation. Do not describe what you plan to do. Do not output anything else.
            """)

    def get_top_eval_score(self) -> int:
        return self.top_eval_score
