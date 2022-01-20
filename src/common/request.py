import json
from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass(frozen=True)
class Request:
    """
    A `Request` specifies how to query a language model (given a prompt,
    complete it).  It is the unified representation for communicating with
    various APIs (e.g., GPT-3, Jurassic).
    """

    # Which model to query
    model: str = "openai/davinci"

    # What prompt do condition the language model on
    prompt: str = ""

    # Temperature parameter that governs diversity
    temperature: float = 1.0

    # Generate this many completions (by sampling from the model)
    num_completions: int = 1

    # Take this many highest probability candidates per token in the completion
    top_k_per_token: int = 1

    # Maximum number of tokens to generate (per completion)
    max_tokens: int = 100

    # Stop generating once we hit one of these strings.
    stop_sequences: List[str] = field(default_factory=list)

    # Should `prompt` be included as a prefix of each completion? (e.g., for
    # evaluating perplexity of the prompt)
    echo_prompt: bool = False

    # Same from tokens that occupy this probability mass (nucleus sampling)
    top_p: float = 1

    # Penalize repetition (OpenAI only)
    presence_penalty: float = 0

    # Penalize repetition (OpenAI only)
    frequency_penalty: float = 0

    # Used to control randomness. Expect different responses for the same
    # request but with different values for `random`.
    random: Optional[str] = None

    @property
    def model_organization(self):
        """Example: 'openai/davinci' => 'openai'"""
        return self.model.split("/")[0]

    @property
    def model_engine(self):
        """Example: 'openai/davinci' => 'davinci'"""
        return self.model.split("/")[1]


@dataclass(frozen=True)
class Token:
    """
    A `Token` represents one token position in a `Sequence`, which has the
    chosen `text` as well as the top probabilities under the model.

    Note: (text, logprob) could exist or not exist in `top_logprobs`.
    """

    # Text that was chosen
    text: str

    # Log probability of generating that
    logprob: float

    # text -> log probability of generating that
    top_logprobs: Dict[str, float]

    def render_lines(self) -> List[str]:
        return [
            f"Text: {repr(self.text)} logprob={self.logprob} Top log probabilities: {json.dumps(self.top_logprobs)}",
        ]


@dataclass(frozen=True)
class Sequence:
    """A `Sequence` is a sequence of tokens."""

    # The concatenation of all the tokens
    text: str

    # The sum of the log probabilities of all tokens
    logprob: float

    # The tokens
    tokens: List[Token]

    def __add__(self, other: "Sequence") -> "Sequence":
        return Sequence(self.text + other.text, self.logprob + other.logprob, self.tokens + other.tokens)

    def render_lines(self) -> List[str]:
        result = [
            f"Text: {self.text}",
            f"Log probability: {self.logprob}",
            "Tokens:",
        ]
        for token in self.tokens:
            result.extend(token.render_lines())
        return result


@dataclass(frozen=True)
class RequestResult:
    """What comes back due to a `Request`."""

    # Whether the request was successful
    success: bool

    # List of completion
    completions: List[Sequence]

    # Whether the query was actually cached
    cached: bool

    # How long did the query take?
    request_time: Optional[float] = None

    # If `success` is false, what was the error?
    error: Optional[str] = None

    def render_lines(self) -> List[str]:
        output = [
            f"Success: {self.success}",
            f"Cached: {self.cached}",
        ]
        if self.request_time:
            output.append(f"Request time: {self.request_time}")
        if self.error:
            output.append(f"Error: {self.error}")

        output.append("Completions")
        for completion in self.completions:
            output.extend(completion.render_lines())
            output.append("")

        return output
