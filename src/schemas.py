from dataclasses import dataclass, field
from typing import List, Optional, Dict
from models import Model

@dataclass
class Request:
    """A request is a concrete query that we make to a client."""
    model: str = 'openai/davinci'
    prompt: str = ''
    temperature: float = 1.0
    numSamples: int = 1  # Generate this many independent samples
    topK: int = 1 # Generate a k-best
    maxTokens: int = 100
    stopSequences: List[str] = field(default_factory=list)

    # For OpenAI's API
    topP: float = 1  # Enable nucleus sampling
    presencePenalty: float = 0
    frequencyPenalty: float = 0

    def model_organization(self):
        """Example: 'openai/davinci' => 'openai'"""
        return self.model.split('/')[0]

    def model_engine(self):
        """Example: 'openai/davinci' => 'davinci'"""
        return self.model.split('/')[1]

@dataclass
class Completion:
    """Represents one result from the API."""
    text: str


@dataclass
class RequestResult:
    """What comes back due to a request."""
    success: bool
    completions: List[Completion]
    cached: bool
    requestTime: Optional[float] = None
    error: Optional[str] = None


@dataclass
class Query:
    """
    A query consists of a `prompt` string, and `settings` (HOCON encoding of things like `temperature`).
    Both can have variables (e.g., ${model}) which can be filled in by `environments`.
    `environments` is a HOCON encoding of a mapping from variables (e.g., model) to a list of arguments.
    """
    prompt: str
    settings: str
    environments: str

@dataclass
class QueryResult:
    """A query produces a list of requests."""
    requests: List[Request]


@dataclass
class GeneralInfo:
    version: str
    exampleQueries: List[Query]
    allModels: List[Model]
