import os
import time
from schemas import Request, RequestResult, Query, QueryResult
from typing import Callable, Any


class Client:
    def expand_query(self, query: Query) -> QueryResult:
        raise NotImplementedError

    def make_request(self, request: Request) -> RequestResult:
        raise NotImplementedError

    """
    Utility methods
    """

    @staticmethod
    def wrap_request_time(compute: Callable[[], Any]) -> Callable[[], Any]:
        """Return a version of `compute` that puts `requestTime` into its output."""

        def wrapped_compute():
            start_time = time.time()
            response = compute()
            end_time = time.time()
            response["requestTime"] = end_time - start_time
            return response

        return wrapped_compute

    @staticmethod
    def get_cache_path(name: str):
        return os.path.join("var", f"{name}-cache.jsonl")
