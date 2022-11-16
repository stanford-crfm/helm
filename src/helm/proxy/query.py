from dataclasses import dataclass
from typing import List
from helm.common.request import Request


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
    """When expanded, a query produces a list of `Request`s."""

    requests: List[Request]
