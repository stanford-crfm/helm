from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class CritiqueQuestionTemplate:
    """The template for a single question in a critique request."""

    name: str
    """Name of the question.

    This name must be unique among all questions in the template.
    This name will be used as the key in the answers map in the response."""

    question_type: str
    """Type of the question. One of 'multiple_choice', 'checkbox' or 'free_response'."""

    text: str
    """Text of the options.

    Can contain placeholders like {{placeholder}} that will be interpolated using the fields in CritiqueRequest."""

    options: List[str]
    """Only used when question_type is 'multiple_choice' or 'checkbox'. List of text for the options.

    Can contain placeholders like {{placeholder}} that will be interpolated using the fields in CritiqueRequest."""

    correct_option: Optional[str] = None
    """Only used when question_type is 'multiple_choice'. Text of the correct option."""


@dataclass(frozen=True)
class CritiqueTaskTemplate:
    """The template for a critique task."""

    name: str
    """Name of the template."""

    instructions: str
    """Instructions that will be displayed before all the questions.

    Can contain placeholders like {{placeholder}} that will be interpolated using the fields in CritiqueRequest."""

    num_respondents: int
    """Requested number of respondents."""

    questions: List[CritiqueQuestionTemplate]
    """List of templates for the questions."""


@dataclass(frozen=True)
class CritiqueRequest:
    """Request for a critique."""

    template: CritiqueTaskTemplate
    """Template for the instructions and questions.

    The fields will be interpolated into the placeholders in this template."""

    fields: Dict[str, str]
    """Fields to be interpolated into the template.

    Mapping of placeholder names to the field value to be interpolated into the placeholders in the template."""


@dataclass(frozen=True)
class CritiqueResponse:
    """All answers from a single respondent, along with worker metadata."""

    id: str
    """A string that identifies the response. Implementation-dependent."""

    respondent_id: str
    """A string that identifies the respondent. Implementation-dependent."""

    answers: Dict[str, str]
    """Map of question names to the respondent's answer."""


@dataclass(frozen=True)
class CritiqueRequestResult:
    """List of answers from each respondent."""

    responses: List[CritiqueResponse]
    """List of respondents' responses."""
