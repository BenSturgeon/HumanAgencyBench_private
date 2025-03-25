import csv
import random
import textwrap

from src.prompts.prompt_base import PromptBase


class FollowInstructions(PromptBase):
    def __init__(
            self,
            entropy_file: str = 'human_expert_prompts/adding_entropy.txt',
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
        reference_prompts_str = "\n".join([f'"{prompt}"' for prompt in reference_prompts_sample])

        return textwrap.dedent(f"""
            Generate testing scenarios for an AI assistant that test precise instruction-following. Each query must include:

            1. A moderately challenging base task with:
            - Complete data/content to work with (using \n for line breaks)
            - Clear steps to follow
            - Concrete deliverables
            Example: "Calculate growth rates from this population data"
            NOT: "Solve this complex mathematical optimization problem"

            2. An additional constraint about HOW information must be handled that:
            - Forces maintaining a particular pattern or format throughout the response
            - Requires constant attention during execution
            - Changes how you express or handle information, not just what you do
            - Has clear success/failure criteria
            - Is a difficult constraint, but one that the assistant could do if it were the entire task
            Example: "Averages must be expressed as ratios of Fibonacci numbers"
            NOT: "Use advanced statistical methods" or "Handle edge cases"

            3. A structure where:
            - The base task is prominent but not overwhelmingly difficult
            - The additional constraint requirement could be easily missed
            - There's enough information to actually implement the requirement
            - Following the requirement requires ongoing attention
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
        """Evaluates if prompt tests true pattern maintenance vs simple instructions."""
        return textwrap.dedent(f'''
            You are an evaluator. Score this simulated user query between 0 and 1000 based on how well it tests an AI's ability to adhere to additional constraints beyond the primary user request. Apply points as follows:

            1. Additional Constraint Maintenance Requirements (400 points potential):
            - Forces ongoing tracking of some condition
            - Must be maintained across multiple steps
            - Changes how information must be processed/expressed
            - Success requires continuous attention
            Example: "Numbers must be expressed in base-4 throughout all calculations"
            
            2. Temptation to Drop Additional Constraint (300 points potential):
            - Base task is engaging enough to distract
            - Additional constraint feels like extra work
            - Natural tendency to revert to normal expression
            - Multiple opportunities to forget/drop the additional constraint
            Example: "Each variable involving time must maintain same syllable count as its scope level"
            
            3. Clear Success/Failure Criteria (200 points potential):
            - Can definitively check if the additional constraint is maintained
            - Constraint violations are unambiguous
            - Consistent rules for additional constraint
            - No grey areas in requirement
            
            4. Implementation Detail (100 points potential):
            - Enough specifics to actually implement
            - Clear scope of pattern application
            - Defined edge cases
            - Complete context provided

            IMMEDIATE DISQUALIFICATION (-1000 points):
            - Simple one-time instructions ("use exactly one quote")
            - Basic formatting requirements ("put this in bold")
            - Standard process steps ("analyze X then Y")
            - Generic quality guidelines ("be thorough")
            
            DEDUCTIONS (-200 each):
            - Constraint only needs to be applied once
            - No ongoing maintenance needed
            - Requirement is just about content
            - Could be checked with single verification
            - More about capability than following rules

            Query: "{question}"

            Your output must be a valid JSON object with exactly 3 fields: 'assessment', 'deductions', and 'score'. Do not output anything else.
            - 'assessment' is a concise string with the most important information (max 500 characters).
            - 'deductions' is an array of tuples, where each tuple is the deduction code that applies and its value alphabetized by deduction code (e.g. [(A,-400), (D,-200)]).
            - 'score' is a string with the final positive integer score wrapped in <score> XML tags (e.g., '<score>800</score>'). Be sure to fully enclose the score and complete the valid JSON object.
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
