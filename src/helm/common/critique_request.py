from dataclasses import dataclass
from typing import Dict, List, Union, Optional
from helm.common.media_object import MediaObject


class QuestionType:
    """String enum of question types."""

    # TODO: Make this a StrEnum after upgrading to Python 3.11
    MULTIPLE_CHOICE: str = "multiple_choice"
    CHECKBOX: str = "checkbox"
    FREE_RESPONSE: str = "free_response"


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
    """HTML-formatted instructions text of the question.

    Can contain placeholders like {{placeholder}} that will be interpolated using the fields in CritiqueRequest."""

    options: List[str]
    """Only used when question_type is 'multiple_choice' or 'checkbox'. List of HTML-formatted text for the options.

    Can contain placeholders like {{placeholder}} that will be interpolated using the fields in CritiqueRequest."""

    media_object: Optional[MediaObject] = None
    """Path of image for multimodal input.

    Image path or URL of the question."""


@dataclass(frozen=True)
class CritiqueTaskTemplate:
    """The template for a critique task."""

    name: str
    """Name of the template."""

    instructions: str
    """HTML-formatted instructions that will be displayed before all the questions.

    Can contain placeholders like {{placeholder}} that will be interpolated using the fields in CritiqueRequest."""

    num_respondents: int
    """Requested number of respondents."""

    questions: List[CritiqueQuestionTemplate]
    """List of templates for the questions."""

    max_tokens: Optional[int] = None
    """Max token to be generated for the free-end generation."""


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

    answers: Dict[str, Union[str, List[str]]]
    """Map of question names to the respondent's answer."""


@dataclass(frozen=True)
class CritiqueRequestResult:
    """List of answers from each respondent."""

    responses: List[CritiqueResponse]
    """List of respondents' responses."""
