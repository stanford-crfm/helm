from dataclasses import dataclass
from typing import Optional


DEFAULT_CLIP_SCORE_MODEL = "openai/clip-vit-large-patch14"


@dataclass(frozen=True)
class CLIPScoreRequest:
    """
    Computes a CLIPScore for a given caption and image.
    """

    # Caption to compute CLIPScore for
    caption: str

    # Location of the image
    image_location: str

    # Which CLIP model to use
    model: str = DEFAULT_CLIP_SCORE_MODEL

    # Compute multilingual CLIPScore
    multilingual: bool = False


@dataclass(frozen=True)
class CLIPScoreResult:
    """Result after sending a `CLIPScoreRequest`."""

    # Whether the request was successful
    success: bool

    # Whether the request was cached
    cached: bool

    # The CLIPScore
    score: float = 0.0

    # If `success` is false, what was the error?
    error: Optional[str] = None
