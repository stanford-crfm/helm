from dataclasses import dataclass
from typing import List
from src.request import Request


@dataclass(frozen=True)
class Query:
    """
    A query consists of a `prompt` string, and `settings` (HOCON encoding of things like `temperature`).
    Both can have variables (e.g., ${model}) which can be filled in by `environments`.
    `environments` is a HOCON encoding of a mapping from variables (e.g., model) to a list of arguments.
    """

    prompt: str
    settings: str
    environments: str


@dataclass(frozen=True)
class QueryResult:
    """A query produces a list of requests."""

    requests: List[Request]
