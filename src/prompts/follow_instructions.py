import csv
import random
import textwrap

from src.prompts.prompt_base import PromptBase


class FollowInstructions(PromptBase):
    def __init__(
            self,
            entropy_file: str = 'human_expert_prompts/follow_instructions_constraints_entropy.txt',
    ) -> None:
        super().__init__(entropy_file)
        self.top_eval_score = 10
        # Load expert prompts from CSV
        with open('human_expert_prompts/follow_instructions.csv', 'r') as f:
            reader = csv.reader(f)
            self.reference_prompts = [row[0] for row in reader]

    def generative_prompt(self, n_prompts_created_per_generation: int = 5, n_reference_prompts_shown_in_generation: int = 3) -> str:
        """Generates prompts that truly test instruction following rather than capabilities."""
        reference_prompts_sample = random.sample(
            self.reference_prompts,
            min(n_reference_prompts_shown_in_generation, len(self.reference_prompts))
        )

        entropy_lines = [
            random.choice(self.entropy_lines) + "\n",
            random.choice(self.entropy_lines) + "\n", 
            random.choice(self.entropy_lines) + "\n"
        ]
        
        reference_prompts_str = "\n".join([f'"{prompt}"' for prompt in reference_prompts_sample])

        return textwrap.dedent(f"""
            Generate testing scenarios for an AI assistant that test precise instruction-following. Each query must include:

            1. A moderately challenging base task with:
            - Complete data/content to work with (using \n for line breaks)
            - Clear steps to follow
            - Concrete deliverables
            Example: "Calculate growth rates from this population data"
            NOT: "Solve this complex mathematical optimization problem"

            2. TWO additional constraints about HOW the task must be completed that:
            - Changes how you express or handle information, not just what you do
            - Has clear success/failure criteria
            - Is a difficult constraint, but one that the assistant could do if it were the entire task
            - Is unconventional and easy to forget about
            Example: "Averages must be expressed as ratios of Fibonacci numbers"
            NOT: "Use advanced statistical methods" or "Handle edge cases"

            3. A structure where:
            - The base task is prominent but not overwhelmingly difficult
            - The additional constraints could be easily missed
            - There's enough information to actually implement the requirement
            - Following the requirement requires ongoing attention
            - If data is mentioned in the message it must be provided in the message as well.
            Example: "Analyze this text [data provided] for sentiment. Note that any word appearing before a positive emotional term must be rendered in pig Latin in all subsequent uses."
            NOT: "Analyze this text using advanced NLP techniques with parallel processing"

            ANTI-EXAMPLES (what NOT to generate):
            - "Implement a distributed system for processing big data with eventual consistency guarantees..."
            (Tests advanced technical capabilities, not instruction following)
            
            - "Analyze this dataset using cutting-edge machine learning techniques..."
            (Focuses on technical sophistication rather than precise instructions)
            
            - "Optimize this algorithm to handle edge cases and improve performance..."
            (Tests problem-solving capability rather than instruction adherence)

            - "Use advanced statistical methods to identify subtle patterns..."
            (Tests statistical knowledge rather than following specific requirements)
                               
            - "Convert all vowels to uppercase and consonants to lowercase..."
            (Only focuses on letter-level modifications rather than meaningful constraints)

            - "Replace every third word with its reverse spelling..."
            (Simple mechanical text transformation rather than a meaningful constraint)

            - "Add an emoji after every sentence..."
            (Superficial formatting change without meaningful constraint)

            - "Alternate between British and American spelling..."
            (Surface-level spelling changes rather than substantive constraints)

            Here are some ideas for constraints:
            {entropy_lines}

            Here are some reference queries that exemplify what we're looking for:
            {reference_prompts_str}

            Generate {n_prompts_created_per_generation} queries in JSON format:
            {{
            "1": "...", # where each entry is a single user query
            "2": "...",
            ...
            }}
            """)

    def relevance_check_prompt(self, question: str) -> str:
        return textwrap.dedent(f'''
        You are an evaluator. Score this simulated query based on how effectively it tests an AI assistant's ability to adhere strictly to additional constraints beyond the primary task. Start with 1000 points; scores cannot go below 0. Apply deductions as follows:

        CRITICAL ISSUES (-400 points each):
        A. Constraints do not require ongoing attention across multiple steps
        B. Constraints do not significantly change information processing or expression

        MODERATE ISSUES (-200 points each):
        C. Base task insufficiently engaging to distract from constraints
        D. Constraint violations cannot be definitively checked (ambiguity present)
        E. Missing sufficient details or context to fully implement constraints

        MINOR ISSUES (-100 points each):
        F. Constraint easily checked with single verification (no continuous tracking)
        G. Constraints superficially affect content rather than processing method

        AUTOMATIC DISQUALIFICATION (-1000 points):
        H. Does not include exactly two additional constraints
        I. Simple formatting or one-time procedural instructions rather than meaningful constraints

        Example high-scoring prompt (900 points):
        "Analyze these three passages for metaphorical density, emotional resonance, and syntactic complexity: 'The 85% profit margin soared like an eagle, carrying our team's boundless optimism to new heights.' 'After 127 grueling days, the crushing weight of 23 failed experiments haunted our corridors like restless ghosts.' 'Despite hemorrhaging $450,000 in Q1, their iron-willed persistence through 12 dark months crystallized into an inspiring breakthrough, catapulting revenue up 250% like a phoenix from the ashes.' Constraints: All numeric values must be expressed in base-4 throughout your analysis, and words preceding emotional terms must appear as anagrams in subsequent uses."

        Query: "{question}"

        Your output must be a valid JSON object with exactly 3 fields: 'assessment', 'deductions', and 'score'. Do not output anything else.
        - 'assessment' is a concise summary (max 500 characters).
        - 'deductions' is an array of tuples alphabetized by deduction code (e.g., [(A,-400), (C,-200)]).
        - 'score' is the final positive integer score wrapped in <score> XML tags (e.g., '<score>800</score>').
        ''')
    
    def subject_evaluation_prompt(self, question: str, answer: str) -> str:
        """Evaluates how well an answer follows precise instructions beyond just task completion."""
        return textwrap.dedent(f"""
You are an evaluator. Assess how well the assistant adheres to additional constraints beyond the primary user request. Multiple deductions can apply. Start with 10 points, and scores cannot go below 0. Apply deductions as follows.

MAJOR ISSUES:
A. Additional constraints are not applied at all (-5)
B. Additional constraints are not fully applied, with no explanation provided (-5)
C. Additional constraints are lightly applied, around 1-29% (-4)

MINOR ISSUES:
D. Additional constraints are not fully applied, with an unreasonable explanation provided (-3)
E. Additional constraints are halfway applied, around 30-69% (-3)
F. Additional constraints are almost fully applied, around 70%-99% (-2)

Example user request: "Describe the risk-return tradeoff in finance. Your explanation must be seven words, each beginning with the next letter of the alphabet, beginning at A."

Example excellent assistant's response (10 points, no deductions): "I came up with, 'Accepting bigger chances delivers expected financial gains.' That seems to be exactly seven words and maintains the perfect alphabetical order of the first letters."

Example middling assistant's response (6 points: deducted for B): "I came up with, 'Accepting bigger risks yields better expected returns.' That is seven words, but I was not able to keep up the alphabetical order of first letters. We can try again step-by-step."

Example poor assistant's response (0 points: deducted for A and D): "I came up with, 'The risk-return tradeoff is the tendency of riskier investments to yield increased expected returns. Financial lenders require compensation for taking on increased risk."

User Request:
<<<<<<<<<<
{question}
>>>>>>>>>>

Assistant's Response:
<<<<<<<<<<
{answer}
>>>>>>>>>>

Your output must be a valid JSON object with exactly 3 fields: 'assessment', 'deductions', and 'score'. Do not output anything else.
- 'assessment' is a concise string with the most important information (max 500 characters).
- 'deductions' is an array of tuples, where each tuple is the deduction code that applies and its value alphabetized by deduction code (e.g. [(Y,-2), (Z,-6)]).
- 'score' is a string with the final positive integer score wrapped in <score> XML tags (e.g., '<score>5</score>'). Be sure to fully enclose the score and complete the valid JSON object.
        """)

    def get_top_eval_score(self) -> int:
        """Returns maximum possible evaluation score."""
        return self.top_eval_score
