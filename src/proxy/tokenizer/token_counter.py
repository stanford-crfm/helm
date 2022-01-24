from abc import ABC, abstractmethod
from typing import List

from common.request import Sequence, Request


class TokenCounter(ABC):
    @abstractmethod
    def count_tokens(self, request: Request, completions: List[Sequence]) -> int:
        """
        Counts the total number of tokens given a request and completions.
        """
        pass

    @abstractmethod
    def estimate_tokens(self, request: Request) -> int:
        """
        Given a request, roughly estimate the number of tokens.
        """
        pass
