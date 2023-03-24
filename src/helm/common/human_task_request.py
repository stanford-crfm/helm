from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class HumanQuestionTemplate:
    """The template for a single question in a human evaluation task."""

    question_type: str
    """Type of the question. One of 'multiple_choice', 'checkbox' or 'free_response'."""

    text: str
    """Text of the options.

    Can contain template tags like {{template_tag}} that will be interpolated using the fields in HumanTaskRequest."""

    options: List[str]
    """Only used when question_type is 'multiple_choice' or 'checkbox'. List of text for the options.

    Can contain template tags like {{template_tag}} that will be interpolated using the fields in HumanTaskRequest."""


@dataclass(frozen=True)
class HumanTaskTemplate:
    """The template for a human evaluation task."""

    name: str
    """Name of the project that will contain all the tasks for the HumanTaskRequest."""

    instructions: str
    """Instructions that will be displayed before all the questions.

    Can contain template tags like {{template_tag}} that will be interpolated using the fields in HumanTaskRequest."""

    num_workers: int
    """Number of requested workers for each task."""

    questions: List[HumanQuestionTemplate]
    """List of templates for the questions."""


@dataclass(frozen=True)
class HumanTaskRequest:
    """Send a request for ."""

    template: HumanTaskTemplate
    fields: Dict[str, str]


@dataclass(frozen=True)
class HumanTaskWorkerResponse:
    """The answers from a worker, along with worker metadata."""

    answers: List[str]
    """The answers from the worker."""

    # TODO: Add worker metadata e.g. ID


@dataclass(frozen=True)
class HumanTaskRequestResult:
    """List of answers ."""

    workers: List[HumanTaskWorkerResponse]
