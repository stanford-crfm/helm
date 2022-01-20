from abc import ABC, abstractmethod
import time
from typing import Callable, Any, Dict

from common.request import Request, RequestResult


class Client(ABC):
    @staticmethod
    def make_cache_key(raw_request: Dict, request: Request) -> Dict:
        """
        Construct the key for the cache using the raw request.
        Add `request.random` to the key, if defined.
        """
        if request.random is not None:
            assert "random" not in raw_request
            cache_key = {**raw_request, "random": request.random}
        else:
            cache_key = raw_request
        return cache_key

    @abstractmethod
    def make_request(self, request: Request) -> RequestResult:
        pass


def wrap_request_time(compute: Callable[[], Any]) -> Callable[[], Any]:
    """Return a version of `compute` that puts `request_time` into its output."""

    def wrapped_compute():
        start_time = time.time()
        response = compute()
        end_time = time.time()
        response["request_time"] = end_time - start_time
        return response

    return wrapped_compute
