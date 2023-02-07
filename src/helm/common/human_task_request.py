from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class HumanQuestionTemplate:
    question_type: str  # multiple_choice, checkbox, or free_response
    text: str
    options: List[str]


@dataclass(frozen=True)
class HumanTaskTemplate:
    name: str
    instructions: str  # Can be HTML
    num_workers: int
    questions: List[HumanQuestionTemplate]


@dataclass(frozen=True)
class HumanTaskRequest:
    template: HumanTaskTemplate
    fields: Dict[str, str]


@dataclass(frozen=True)
class HumanTaskRequestResult:
    answers: List[List[str]]
