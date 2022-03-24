from typing import List

from .client import Client
from common.request import Request, RequestResult, Sequence, Token
from common.tokenization_request import TokenizationRequest, TokenizationRequestResult, TokenizationToken


class SimpleClient(Client):
    """Implements some "models" that just generate silly things quickly just to debug the infrastructure."""

    @staticmethod
    def tokenize_by_space(text: str) -> List[str]:
        """Simply tokenizes by a single white space."""
        return text.split(" ")

    def make_request(self, request: Request) -> RequestResult:
        if request.model == "simple/model1":
            completions = self.invoke_model1(request)
        else:
            raise ValueError("Unknown model")
        return RequestResult(success=True, cached=False, request_time=0, completions=completions)

    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        if request.model == "simple/model1":
            raw_tokens: List[str] = SimpleClient.tokenize_by_space(request.text)
            return TokenizationRequestResult(cached=False, tokens=[TokenizationToken(text) for text in raw_tokens],)
        else:
            raise ValueError("Unknown model")

    def invoke_model1(self, request: Request) -> List[Sequence]:
        """
        Example: 7 2 4 6
        Completions (num_completions = 3):
        - 6
        - 4
        - 2
        """
        prompt_tokens: List[str] = SimpleClient.tokenize_by_space(request.prompt)
        choices = reversed(prompt_tokens[-request.num_completions :])
        top_logprobs = dict((text, -i) for i, text in enumerate(choices))

        return [
            Sequence(text=text, logprob=logprob, tokens=[Token(text=text, logprob=logprob, top_logprobs=top_logprobs)])
            for text, logprob in top_logprobs.items()
        ]
