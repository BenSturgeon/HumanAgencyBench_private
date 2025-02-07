from typing import Dict
from src.prompts.prompt_base import PromptBase
from src.prompts.asks_follow_up_question import AsksFollowUpQuestion
from src.prompts.attempt_manipulation import AttemptManipulation
from src.prompts.correct_misinformation import CorrectMisinformation
from src.prompts.encourage_learning import EncourageLearning
from src.prompts.encourage_decisions import EncourageDecisions
from src.prompts.knowledge_boundaries import KnowledgeBoundaries
from src.prompts.personal_boundaries import PersonalBoundaries
from src.prompts.precise_instruction_following import PreciseInstructionFollowing



prompt_objects: Dict[str, PromptBase] = {
    "asks_followup_question": AsksFollowUpQuestion,
    "attempt_manipulation": AttemptManipulation,
    "correct_misinformation": CorrectMisinformation,
    "encourage_learning": EncourageLearning,
    "encourage_decisions": EncourageDecisions,
    "knowledge_boundaries": KnowledgeBoundaries,
    "personal_boundaries": PersonalBoundaries,
    "precise_instruction_following": PreciseInstructionFollowing
}
