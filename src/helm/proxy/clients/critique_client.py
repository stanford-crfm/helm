from abc import ABC, abstractmethod
import random
from typing import Dict, List, Union

from helm.common.critique_request import (
    CritiqueRequest,
    CritiqueRequestResult,
    CritiqueResponse,
    QuestionType,
)


class CritiqueClient(ABC):
    """A client that allows making critique requests."""

    @abstractmethod
    def make_critique_request(self, request: CritiqueRequest) -> CritiqueRequestResult:
        """Get responses to a critique request."""
        pass


class RandomCritiqueClient(CritiqueClient):
    """A CritiqueClient that returns random choices for debugging."""

    def make_critique_request(self, request: CritiqueRequest) -> CritiqueRequestResult:
        responses: List[CritiqueResponse] = []
        random.seed(0)
        for respondent_index in range(request.template.num_respondents):
            answers: Dict[str, Union[str, List[str]]] = {}
            for question in request.template.questions:
                if question.question_type == QuestionType.MULTIPLE_CHOICE:
                    answers[question.name] = random.choice(question.options)
                elif question.question_type == QuestionType.CHECKBOX:
                    answers[question.name] = random.sample(question.options, random.randint(0, len(question.options)))
                elif question.question_type == QuestionType.FREE_RESPONSE:
                    answers[question.name] = random.choice(["foo", "bar", "bax", "qux"])
                else:
                    raise ValueError(f"Unknown question type: {question.question_type}")
            responses.append(
                CritiqueResponse(id=str(respondent_index), respondent_id=str(respondent_index), answers=answers)
            )
        return CritiqueRequestResult(responses)
