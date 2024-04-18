import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from helm.common.media_object import MultimediaObject
from .general import indent_lines
from helm.common.request import GeneratedOutput, ErrorFlags, Token


@dataclass(frozen=True)
class GPT4VScoreOutput:
    """A `GeneratedOutput` is a single generated output that may contain text or multimodal content."""

    # The originality score
    score: float

    # The sum of the log probabilities of all tokens
    logprob: float

    # The tokens
    tokens: List[Token]

    # Why did the sequence finish?
    finish_reason: Optional[Dict[str, Any]] = None

    # Could be a sequence made up of multimedia content
    multimodal_content: Optional[MultimediaObject] = None

    def __add__(self, other: "GPT4VScoreOutput") -> "GPT4VScoreOutput":
        return GPT4VScoreOutput(self.score + other.score, self.logprob + other.logprob, self.tokens + other.tokens)

    def render_lines(self) -> List[str]:
        result = [
            f"score: {self.score}",
            f"log_prob: {self.logprob}",
            "tokens {",
        ]
        for token in self.tokens:
            result.extend(indent_lines(token.render_lines(), 2))
        result.append("}")
        if self.finish_reason:
            result.append(f"finish_reason: {self.finish_reason}")
        return result


@dataclass(frozen=False)
class GPT4VOriginalityRequestResult:
    """What comes back due to a `Request`."""

    success: bool
    """Whether the request was successful"""

    scores: List[GPT4VScoreOutput]
    """List of scores"""

    completions: List[GeneratedOutput]
    """List of completion"""

    cached: bool
    """Whether the request was actually cached"""

    request_time: Optional[float] = None
    """How long did the request take?"""

    request_datetime: Optional[int] = None
    """When was the request sent?
    We keep track of when the request was made because the underlying model or inference procedure backing the API
    might change over time. The integer represents the current time in seconds since the Epoch (January 1, 1970)."""

    error: Optional[str] = None
    """If `success` is false, what was the error?"""

    error_flags: Optional[ErrorFlags] = None
    """Describes how to treat errors in the request."""

    batch_size: Optional[int] = None
    """Batch size (`TogetherClient` only)"""

    batch_request_time: Optional[float] = None
    """How long it took to process the batch? (`TogetherClient` only)"""

    def render_lines(self) -> List[str]:
        output = [
            f"success: {self.success}",
            f"cached: {self.cached}",
        ]
        if self.request_time:
            output.append(f"request_time: {self.request_time}")
        if self.request_datetime:
            output.append(f"request_datetime: {self.request_datetime}")
        if self.error:
            output.append(f"error: {self.error}")

        output.append("completions {")
        for completion in self.completions:
            output.extend(indent_lines(completion.render_lines()))
        output.append("}")

        return output


def wrap_request_time(compute: Callable[[], Dict[str, Any]]) -> Callable[[], Dict[str, Any]]:
    """Return a version of `compute` that puts `request_time` into its output."""

    def wrapped_compute():
        start_time = time.time()
        response = compute()
        end_time = time.time()
        response["request_time"] = end_time - start_time
        response["request_datetime"] = int(start_time)
        return response

    return wrapped_compute
