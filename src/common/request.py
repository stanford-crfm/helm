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

    def __str__(self) -> str:
        return (
            f"Model: {self.model}\n"
            f"Temperature: {self.temperature}\n"
            f"Number of completions: {self.num_completions}\n"
            f"Top k per token: {self.top_k_per_token}\n"
            f"Maximum number of tokens: {self.max_tokens}\n"
            f"Stop sequences: {', '.join(self.stop_sequences)}\n"
            f"Echo prompt: {self.echo_prompt}\n"
            f"Top p: {self.top_p}\n"
            f"Presence penalty: {self.presence_penalty}\n"
            f"Frequency penalty: {self.frequency_penalty}\n"
            f"Prompt: {self.prompt}\n"
        )

    @property
    def model_organization(self):
        """Example: 'openai/davinci' => 'openai'"""
        return self.model.split("/")[0]

    @property
    def model_engine(self):
        """Example: 'openai/davinci' => 'davinci'"""
        return self.model.split("/")[1]

    def to_dict(self) -> Dict:
        return {
            "model": self.model,
            "prompt": self.prompt,
            "temperature": self.temperature,
            "num_completions": self.num_completions,
            "top_k_per_tokens": self.top_k_per_token,
            "max_tokens": self.max_tokens,
            "stop_sequences": self.stop_sequences,
            "echo_prompt": self.echo_prompt,
            "top_p": self.top_p,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
        }


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

    def __str__(self) -> str:
        return (
            f"Text: {self.text}\nLog probability: {self.logprob}\n"
            f"Top log probabilities: {json.dumps(self.top_logprobs)}"
        )

    def to_dict(self) -> Dict:
        return {"text": self.text, "logprob": self.logprob, "top_logprobs": self.top_logprobs}


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

    def __str__(self) -> str:
        tokens: str = "\n".join(str(token) for token in self.tokens)
        return f"Text: {self.text}\nLog probability: {self.logprob}\nTokens:\n{tokens}"

    def to_dict(self) -> Dict:
        return {"text": self.text, "logprob": self.logprob, "tokens": [token.to_dict() for token in self.tokens]}


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

    def __str__(self) -> str:
        completions: str = "\n\n".join(str(completion) for completion in self.completions)
        output: str = f"Success: {self.success}\nCached: {self.cached}\n"
        if self.request_time:
            output += f"Request time: {self.request_time}"
        if self.error:
            output += f"Error: {self.error}"
        output += f"\nCompletions:\n{completions}"
        return output

    def to_dict(self) -> Dict:
        result = {
            "success": self.success,
            "completions": [completion.to_dict() for completion in self.completions],
            "cached": self.cached,
        }
        if self.request_time:
            result["request_time"] = self.request_time
        if self.error:
            result["error"] = self.error
        return result
