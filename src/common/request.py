from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass(frozen=True)
class Request:
    """A request is a concrete query that we make to a client."""

    model: str = "openai/davinci"
    prompt: str = ""
    temperature: float = 1.0
    num_completions: int = 1  # Generate this many completions
    top_k_per_token: int = 1  # Take this many highest probability candidates per token in the completion
    max_tokens: int = 100  # Maximum number of tokens to generate
    stop_sequences: List[str] = field(default_factory=list)
    echo_prompt: bool = False  # `prompt` should be prefix of each completion? (for evaluating perplexity)

    # For OpenAI's API
    top_p: float = 1  # Enable nucleus sampling
    presence_penalty: float = 0
    frequency_penalty: float = 0

    def model_organization(self):
        """Example: 'openai/davinci' => 'openai'"""
        return self.model.split("/")[0]

    def model_engine(self):
        """Example: 'openai/davinci' => 'davinci'"""
        return self.model.split("/")[1]


@dataclass(frozen=True)
class Token:
    """
    A `Token` represents one token position in the sequence, which has the
    chosen `text` as well as the top probabilities under the model.
    """

    text: str  # Text that was chosen
    log_prob: float  # Log probability of that text
    top_choices: Dict[str, float]  # text -> log_prob


@dataclass(frozen=True)
class Sequence:
    """A `Sequence` is a sequence of tokens."""

    text: str
    log_prob: float
    tokens: List[Token]


@dataclass(frozen=True)
class RequestResult:
    """What comes back due to a `Request`."""

    success: bool
    completions: List[Sequence]
    cached: bool
    request_time: Optional[float] = None
    error: Optional[str] = None
