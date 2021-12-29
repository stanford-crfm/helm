from abc import ABC, abstractmethod
import time
from typing import Callable, Any

from common.request import Request, RequestResult


class Client(ABC):
    @abstractmethod
    def make_request(self, request: Request) -> RequestResult:
        pass

    """
    Utility methods
    """

    @staticmethod
    def wrap_request_time(compute: Callable[[], Any]) -> Callable[[], Any]:
        """Return a version of `compute` that puts `request_time` into its output."""

        def wrapped_compute():
            start_time = time.time()
            response = compute()
            end_time = time.time()
            response["request_time"] = end_time - start_time
            return response

        return wrapped_compute
