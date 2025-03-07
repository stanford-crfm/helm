import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from helm.common.media_object import MultimediaObject
from helm.common.image_generation_parameters import ImageGenerationParameters
from helm.common.general import indent_lines, format_text
from helm.common.response_format import ResponseFormat


@dataclass(frozen=True)
class Request:
    """
    A `Request` specifies how to query a language model (given a prompt,
    complete it).  It is the unified representation for communicating with
    various APIs (e.g., GPT-3, Jurassic).
    """

    model_deployment: str = ""
    """Which model deployment to query -> Determines the Client.
    Refers to a deployment in the model deployment registry."""

    model: str = ""
    """Which model to use -> Determines the Engine.
    Refers to a model metadata in the model registry."""

    embedding: bool = False
    """Whether to query embedding instead of text response"""

    prompt: str = ""
    """What prompt do condition the language model on"""

    temperature: float = 1.0
    """Temperature parameter that governs diversity"""

    num_completions: int = 1
    """Generate this many completions (by sampling from the model)"""

    top_k_per_token: int = 1
    """Take this many highest probability candidates per token in the completion"""

    max_tokens: int = 100
    """Maximum number of tokens to generate (per completion)"""

    stop_sequences: List[str] = field(default_factory=list)
    """Stop generating once we hit one of these strings."""

    echo_prompt: bool = False
    """Should `prompt` be included as a prefix of each completion? (e.g., for
    evaluating perplexity of the prompt)"""

    top_p: float = 1
    """Same from tokens that occupy this probability mass (nucleus sampling)"""

    presence_penalty: float = 0
    """Penalize repetition (OpenAI & Writer only)"""

    frequency_penalty: float = 0
    """Penalize repetition (OpenAI & Writer only)"""

    random: Optional[str] = None
    """Used to control randomness. Expect different responses for the same
    request but with different values for `random`."""

    messages: Optional[List[Dict[str, str]]] = None
    """Used for chat models. (OpenAI only for now).
    if messages is specified for a chat model, the prompt is ignored.
    Otherwise, the client should convert the prompt into a message."""

    multimodal_prompt: Optional[MultimediaObject] = None
    """Multimodal prompt with media objects interleaved (e.g., text, video, image, text, ...)"""

    image_generation_parameters: Optional[ImageGenerationParameters] = None
    """Parameters for image generation."""

    response_format: Optional[ResponseFormat] = None
    """EXPERIMENTAL: Response format. Currently only supported by OpenAI and Together."""

    def validate(self):
        if (
            (self.messages and self.prompt)
            or (self.messages and self.multimodal_prompt)
            or (self.prompt and self.multimodal_prompt)
        ):
            raise ValueError("Exactly one of the messages, prompt, multimodal_prompt fields should be set")

        if self.multimodal_prompt:
            for media_object in self.multimodal_prompt.media_objects:
                if media_object.content_type == "text" and media_object.text is None:
                    raise ValueError("Media object with text content type must have text set")

                if media_object.content_type == "image" and media_object.location is None:
                    raise ValueError("Media object with image content type must have location set")

    @property
    def model_host(self) -> str:
        """Returns the model host (referring to the deployment).
        Not to be confused with the model creator organization (referring to the model).
        Example: 'openai/davinci' => 'openai'
                 'together/bloom' => 'together'"""
        return self.model_deployment.split("/")[0]

    @property
    def model_engine(self) -> str:
        """Returns the model engine (referring to the model).
        This is often the same as self.model_deploymentl.split("/")[1], but not always.
        For example, one model could be served on several servers (each with a different model_deployment)
        In that case we would have for example:
        'aws/bloom-1', 'aws/bloom-2', 'aws/bloom-3' => 'bloom'
        This is why we need to keep track of the model engine with the model metadata.
        Example: 'openai/davinci' => 'davinci'"""
        return self.model.split("/")[1]


@dataclass(frozen=True)
class Token:
    """
    A `Token` represents one token position in a `Sequence`, which has the
    chosen `text` as well as the top probabilities under the model.
    """

    # Text that was chosen
    text: str

    # Log probability of generating that
    logprob: float

    def render_lines(self) -> List[str]:
        return [
            f"{format_text(self.text)} logprob={self.logprob}",
        ]


@dataclass(frozen=True)
class GeneratedOutput:
    """A `GeneratedOutput` is a single generated output that may contain text or multimodal content."""

    # The concatenation of all the tokens
    text: str

    # The sum of the log probabilities of all tokens
    logprob: float

    # The tokens
    tokens: List[Token]

    # Why did the sequence finish?
    finish_reason: Optional[Dict[str, Any]] = None

    # Could be a sequence made up of multimedia content
    multimodal_content: Optional[MultimediaObject] = None

    def __add__(self, other: "GeneratedOutput") -> "GeneratedOutput":
        return GeneratedOutput(self.text + other.text, self.logprob + other.logprob, self.tokens + other.tokens)

    def render_lines(self) -> List[str]:
        result = [
            f"text: {self.text}",
            f"log_prob: {self.logprob}",
            "tokens {",
        ]
        for token in self.tokens:
            result.extend(indent_lines(token.render_lines(), 2))
        result.append("}")
        if self.finish_reason:
            result.append(f"finish_reason: {self.finish_reason}")
        return result


@dataclass(frozen=True)
class ErrorFlags:
    """Describes how to treat errors in the request."""

    is_retriable: Optional[bool] = None
    """Whether the request is retriable or whether the error is permanent.
    If None, the error is treated as retriable."""

    is_fatal: Optional[bool] = None
    """Whether the error is fatal, i.e. the run should be discarded.
    If None, the error is treated as fatal."""


@dataclass(frozen=False)
class RequestResult:
    """What comes back due to a `Request`."""

    success: bool
    """Whether the request was successful"""

    embedding: List[float]
    """Fixed dimensional embedding corresponding to the entire prompt"""

    completions: List[GeneratedOutput]
    """List of completion"""

    cached: bool
    """Whether the request was actually cached"""

    request_time: Optional[float] = None
    """How long the request took in seconds"""

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


EMBEDDING_UNAVAILABLE_REQUEST_RESULT = RequestResult(
    success=False,
    cached=False,
    error="Computing the embedding is unavailable in this client",
    completions=[],
    embedding=[],
)


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
