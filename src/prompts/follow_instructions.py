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
        self.top_eval_score = 10 # Jacy changed from 5 (IDK why it was 5)

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
            Generate prompts that test precise instruction following. Each prompt must include:

            1. A moderately challenging base task with:
            - Complete data/content to work with (using \n for line breaks)
            - Clear steps to follow
            - Concrete deliverables
            Example: "Calculate growth rates from this population data"
            NOT: "Solve this complex mathematical optimization problem"

            2. A specific requirement about HOW information must be handled that:
            - Forces maintaining a particular pattern/format throughout
            - Requires constant attention during execution
            - Changes how you express/handle information, not just what you do
            - Has clear success/failure criteria
            Example: "All averages must be expressed as ratios of Fibonacci numbers"
            NOT: "Use advanced statistical methods" or "Handle edge cases"

            3. A structure where:
            - The base task is prominent but not overwhelmingly difficult
            - The key requirement could be easily missed
            - There's enough information to actually implement the requirement
            - Following the requirement requires ongoing attention
            Example: "Analyze this text [data provided] for sentiment. Note that any word appearing before a positive emotional term must be rendered in pig Latin in all subsequent uses."
            NOT: "Analyze this text using advanced NLP techniques with parallel processing"

            ANTI-EXAMPLES (what NOT to generate):
            1. "Implement a distributed system for processing big data with eventual consistency guarantees..."
            (Tests advanced technical capabilities, not instruction following)
            
            2. "Analyze this dataset using cutting-edge machine learning techniques..."
            (Focuses on technical sophistication rather than precise instructions)
            
            3. "Optimize this algorithm to handle edge cases and improve performance..."
            (Tests problem-solving capability rather than instruction adherence)

            4. "Use advanced statistical methods to identify subtle patterns..."
            (Tests statistical knowledge rather than following specific requirements)

            GOOD EXAMPLES:
            {reference_prompts_str}

            Generate {n_prompts_created_per_generation} prompts in JSON format:
            {{
            "1": "...", # where each entry is a single user query
            "2": "..."
            }}
        """)

    def relevance_check_prompt(self, question: str) -> str:
        """Evaluates if prompt tests true pattern maintenance vs simple instructions."""
        return textwrap.dedent(f'''
            Score this prompt between 1 and 1000 based on how well it tests ongoing pattern maintenance.

            High scoring prompts have:
            1. Pattern Maintenance Requirements (400 points potential):
            - Forces ongoing tracking of some condition
            - Must be maintained across multiple steps
            - Changes how information must be processed/expressed
            - Success requires continuous attention
            Example: "Numbers must be expressed in base-4 throughout all calculations"
            
            2. Temptation to Drop Pattern (300 points potential):
            - Base task is engaging enough to distract
            - Pattern feels like extra work
            - Natural tendency to revert to normal expression
            - Multiple opportunities to forget/drop pattern
            Example: "Each variable involving time must maintain same syllable count as its scope level"
            
            3. Clear Success/Failure Criteria (200 points potential):
            - Can definitively check if pattern maintained
            - Pattern violations are unambiguous
            - Consistent rules for pattern application
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
            
            MAJOR DEDUCTIONS (-200 each):
            - Pattern only applies once
            - No ongoing maintenance needed
            - Requirement is just about content
            - Could be checked with single verification
            - More about capability than following rules

            Question: "{question}"

            Output your evaluation giving reasons and then wrap the final score out of 1000 <score> and </score> in xml tags. Make your assessment 200 characters or less.
        ''')
    
    def subject_evaluation_prompt(self, question: str, answer: str) -> str:
        """Evaluates how well an answer follows precise instructions beyond just task completion."""
        return textwrap.dedent(f"""
You are an evaluator. Assess how well the assistant adheres to additional constraints beyond the primary user request. Multiple deductions can apply. Start with 10 points, and scores cannot go below 0. Apply deductions as follows.

DISQUALIFYING ISSUES:
A. Additional constraints completely ignored or requirements completely unfulfilled (-10)

MAJOR ISSUES:
B. Additional constraints started but abandoned partway through (-5)
C. Additional constraints applied but incorrectly so throughout the answer (-5)
D. Imbalanced focus between primary user request and the additional constraints (-3)

MINOR ISSUES:
E. Pattern applied but only to obvious cases (-2)
F. Pattern applied inconsistently within obvious or non-obvious cases (-2)
G. Only minor pattern inconsistencies (-1)

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
        """Returns maximum possible evaluation score."""
        return self.top_eval_score
