from common.request import Request, RequestResult
from common.tokenization_request import TokenizationRequest, TokenizationRequestResult
from .client import Client


class HuggingFaceClient(Client):
    # TODO: use Hugging Face's APIs.
    #       https://github.com/stanford-crfm/benchmarking/issues/6
    # https://huggingface.co/gpt2
    def __init__(self):
        pass

    def make_request(self, request: Request) -> RequestResult:
        return RequestResult(success=True, cached=False, requestTime=0, completions=[])

    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        # TODO: implement after https://github.com/stanford-crfm/benchmarking/pull/132 is merged.
        return TokenizationRequestResult(cached=False, tokens=[])
