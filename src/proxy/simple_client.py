from typing import List

from .client import Client
from common.request import Request, RequestResult, Completion


class SimpleClient(Client):
    """Implements some "models" that just generate silly things quickly just to debug the infrastructure."""

    def make_request(self, request: Request) -> RequestResult:
        if request.model == "simple/model1":
            completions = self.invoke_model1(request)
        else:
            raise ValueError("Unknown model")
        return RequestResult(success=True, cached=False, request_time=0, completions=completions)

    def invoke_model1(self, request: Request) -> List[Completion]:
        """
        Example: 7 2 4 6
        Completions (num_completions = 3):
        - 6
        - 4
        - 2
        """
        tokens = request.prompt.split(" ")
        return [Completion(text=token) for token in reversed(tokens[-request.num_completions :])]
