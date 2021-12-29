from dataclasses import dataclass, field
from typing import List, Optional


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
class Completion:
    """Represents one result from the API."""

    text: str


@dataclass(frozen=True)
class RequestResult:
    """What comes back due to a request."""

    success: bool
    completions: List[Completion]
    cached: bool
    request_time: Optional[float] = None
    error: Optional[str] = None
