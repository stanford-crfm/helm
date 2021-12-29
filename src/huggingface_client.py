from client import Client
from schemas import Request, RequestResult


class HuggingFaceClient(Client):
    # TODO: use Hugging Face's APIs.
    # https://huggingface.co/gpt2
    def __init__(self):
        pass

    def make_request(self, request: Request) -> RequestResult:
        return RequestResult(success=True, cached=False, requestTime=0, completions=[])
