from common.request import Request, RequestResult
from .client import Client


class HuggingFaceClient(Client):
    # TODO: use Hugging Face's APIs.
    #       https://github.com/stanford-crfm/benchmarking/issues/6
    # https://huggingface.co/gpt2
    def __init__(self):
        pass

    def make_request(self, request: Request) -> RequestResult:
        return RequestResult(success=True, cached=False, requestTime=0, completions=[])
