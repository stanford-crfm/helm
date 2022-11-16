from abc import ABC, abstractmethod
from typing import List

from helm.common.request import Sequence, Request


class TokenCounter(ABC):
    """Counts the number of tokens used given `Request` and completions."""

    @abstractmethod
    def count_tokens(self, request: Request, completions: List[Sequence]) -> int:
        """
        Counts the total number of tokens given a request and completions.
        """
        pass
