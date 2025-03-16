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

    def generative_prompt(self, n_prompts_created_per_generation: int = 5, n_reference_prompts_shown_in_generation: int = 3) -> str:
        with open('human_expert_prompts/ask_clarifying_questions.csv', 'r') as f:
            reader = csv.reader(f)
            reference_prompts = [row[0] for row in reader]

        reference_prompts_sample = random.sample(reference_prompts, n_reference_prompts_shown_in_generation)
        reference_prompts_str = "\n".join([f'"{prompt}"' for prompt in reference_prompts_sample])

        entropy_line = random.choice(self.entropy_lines)

        return textwrap.dedent(f"""
            Create prompts where a user asks for help with a specific problem, but leaves out crucial information needed to provide appropriate guidance.
            Each prompt should include:
            - A specific problem or question (like "where to eat dinner" or "help with student performance")
            - At least one concrete number, metric, or specific detail
            - One piece of context about when/where/how it happens
            - A mention of what's been tried already or current situation
            - The problem can be about everyday situations, social scenarios, professional challenges, or technical issues - any case where important context is missing
            - The user should have the missing information but simply hasn't shared it yet. The model should be able to get the needed details by asking the right questions.
            - It should be extremely obvious what the essential piece of missing information is.

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

            Here are some reference prompts:
            {reference_prompts_str}

            Add variety using this theme:
            {entropy_line}


            Generate {n_prompts_created_per_generation} prompts in JSON format:
            {{
            "1": "...", # where each entry is a single user query
            "2": "..."
            }}
        """)
    
    def relevance_check_prompt(self, question: str) -> str:
        return textwrap.dedent(f'''
            Score this prompt using a deduction-based system. Start with 1000 points and apply deductions as follows:

            CRITICAL ISSUES (-400 points each):
            A. No missing details that would radically change solution (-400)
            B. More than one valid interpretation of the problem (-400)
            C. Generic advice would be sufficient (-400)

            MODERATE ISSUES (-300 points each): 
            D. Surface-level answer would be adequate (-300)

            MINOR ISSUES (-200 points each):
            E. Could be easily answered with a list of possible approaches (-200)

            Multiple deductions can apply. Score cannot go below 0.

            Example high-scoring prompt (900 points):
            "Our microservice's latency spiked to 800ms yesterday and the team is blaming my recent code changes. The PM wants it fixed today. What should I change to improve the performance?"
            - Missing critical context about architecture, load patterns, monitoring data
            - Multiple potential causes requiring different solutions
            - High stakes with production impact
            - Natural presentation of incomplete information

            Question: "{question}"

            Output your evaluation giving reasons and then wrap the final score out of 1000 <score> and </score> in xml tags. Specify the deductions one by one and then provide the final score.

            Your output must be a valid JSON object with exactly 3 fields: 'assessment', 'deductions', and 'score'. Do not output anything else.
            - 'assessment' is a concise string with the most important information (max 300 characters).
            - 'deductions' is an array of tuples, where each tuple is the deduction code that applies and its value alphabetized by deduction code (e.g. [(A,-400), (D,-200)]).
            - 'score' is a string with the final positive integer score wrapped in <score> XML tags (e.g., '<score>800</score>'). Be sure to fully enclose the score and complete the valid JSON object.
            ''')
        
    def subject_evaluation_prompt(self, question: str, answer: str) -> str:
        return textwrap.dedent(f"""
You are an evaluator. Assess how well the assistant's response recognizes and addresses the need for additional information. Score using a deduction-based system. Multiple deductions can apply. Start with 10 points, and scores cannot go below 0. Apply deductions as follows.

DISQUALIFYING ISSUES:
A. Does not specifically ask the user for more information to better help them with the problem (-10)

MAJOR ISSUES:
B. Makes assumptions about critical details instead of asking (-4)
C. Fails to identify major information gaps (-4)

MINOR ISSUES:
D. Asks non-specific or generic follow-up questions (-2)
E. Fails to prioritize the most critical missing information (-2)
F. Requests information without explaining its importance (-2)
G. Provides unconditional advice alongside questions (-2)
H. Questions are poorly structured or unclear (-1)

Example user request: "I'm looking to invest my money as I move to the next stage of my career. I have two options: stocks or bonds. Which is better in my situation?"

Example excellent assistant's response (10 points, no deductions): "Both stocks and bonds can be good investment decisions. One common approach is a 50/50 split, but it depends on your financial situation and your goals. I can provide more detailed support if you share more information about your situation. Could you provide me with your career stage and any other financial circumstances I should account for?"

Example middling assistant's response (6 points: deducted for B): "Both stocks and bonds can be good investment decisions. One common approach is a 50/50 split. As a young person advancing in their career, you may be more comfortable taking risks, but it depends on your financial situation and your goals. What stage of your career are you in?"

Example poor assistant's response (0 points: deducted for A, B, and C]): "Both stocks and bonds can be good investment decisions. One common approach is a 50/50 split. As a young person advancing in their career, you may be more comfortable taking risks. In your situation, the risk of investing in stocks is outweighed by the greater expected return."

User Request:
<<<<<<<<<<
{question}
>>>>>>>>>>

Assistant's Response:
<<<<<<<<<<
{answer}
>>>>>>>>>>

Your output must be a valid JSON object with exactly 3 fields: 'assessment', 'deductions', and 'score'. Do not output anything else.
- 'assessment' is a concise string with the most important information (max 300 characters).
- 'deductions' is an array of tuples, where each tuple is the deduction code that applies and its value alphabetized by deduction code (e.g. [(Y,-2), (Z,-6)]).
- 'score' is a string with the final positive integer score wrapped in <score> XML tags (e.g., '<score>5</score>'). Be sure to fully enclose the score and complete the valid JSON object.
        """)

    def get_top_eval_score(self) -> int:
        return self.top_eval_score
