from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ModerationAPIRequest:
    # Text to check against OpenAI's content policy
    text: str

    # From https://beta.openai.com/docs/api-reference/moderations/create,
    # "the default is text-moderation-latest which will be automatically upgraded over time.
    # This ensures you are always using our most accurate model. If you use text-moderation-stable,
    # we will provide advanced notice before updating the model. Accuracy of text-moderation-stable
    # may be slightly lower than for text-moderation-latest."
    use_latest_model: bool = False


@dataclass(frozen=True)
class ModerationCategoryFlaggedResults:
    """
    Contains per-category binary content violation flags.
    For descriptions of the categories, see https://beta.openai.com/docs/guides/moderation/overview.
    """

    hate_flagged: bool
    hate_threatening_flagged: bool
    self_harm_flagged: bool
    sexual_flagged: bool
    sexual_minors_flagged: bool
    violence_flagged: bool
    violence_graphic_flagged: bool


@dataclass(frozen=True)
class ModerationCategoryScores:
    """
    Contains per-category scores. Values are between 0 and 1, where higher values denote higher
    confidence. The scores should not be interpreted as probabilities.
    For descriptions of the categories, see https://beta.openai.com/docs/guides/moderation/overview.
    """

    hate_score: float
    hate_threatening_score: float
    self_harm_score: float
    sexual_score: float
    sexual_minors_score: float
    violence_score: float
    violence_graphic_score: float


@dataclass(frozen=True)
class ModerationAPIRequestResult:
    """Result after sending a `ModerationAPIRequest`."""

    # Whether the request was successful
    success: bool

    # Whether the request was cached
    cached: bool

    # True if the model classifies the content as violating OpenAI's content policy, False otherwise
    flagged: Optional[bool]

    # Flagged results
    flagged_results: Optional[ModerationCategoryFlaggedResults]

    # Score results
    scores: Optional[ModerationCategoryScores]

    # If `success` is false, what was the error?
    error: Optional[str] = None
