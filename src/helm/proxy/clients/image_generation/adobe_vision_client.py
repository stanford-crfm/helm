from typing import List, Dict

from helm.common.cache import Cache, CacheConfig
from helm.common.request import Request, RequestResult, Sequence, TextToImageRequest
from helm.common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
    DecodeRequest,
    DecodeRequestResult,
)

from helm.proxy.clients.client import Client


class AdobeVisionClient(Client):
    """
    Client for Adobe vision models. Offline eval only.
    """

    SUPPORTED_MODELS: List[str] = ["giga-gan", "firefly"]

    @staticmethod
    def convert_to_raw_request(request: Request) -> Dict:
        # Use default hyperparameters for everything else
        raw_request: Dict = {
            "request_type": "image-model-inference",
            "model": request.model_engine,
            "prompt": request.prompt,
            "n": request.num_completions,
        }
        if request.random is not None:
            raw_request["random"] = request.random
        return raw_request

    def __init__(self, cache_config: CacheConfig):
        self._cache = Cache(cache_config)
        self._promptist_model = None
        self._promptist_tokenizer = None

    def make_request(self, request: Request) -> RequestResult:
        if not isinstance(request, TextToImageRequest):
            raise ValueError(f"Wrong type of request: {request}")

        if request.model_engine not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {request.model_engine}")

        raw_request = AdobeVisionClient.convert_to_raw_request(request)
        raw_request.pop("random", None)
        cache_key: Dict = Client.make_cache_key(raw_request, request)

        try:

            def fail():
                raise RuntimeError(
                    f"The result has not been uploaded to the cache for the following request: {cache_key}"
                )

            response, cached = self._cache.get(cache_key, fail)
        except RuntimeError as e:
            error: str = f"Adobe Vision Client error: {e}"
            return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])

        completions: List[Sequence] = [
            Sequence(text="", logprob=0, tokens=[], file_location=file_path) for file_path in response["images"]
        ]
        return RequestResult(
            success=True,
            cached=cached,
            request_time=response["request_time"],
            completions=completions,
            embedding=[],
        )

    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        raise NotImplementedError("This client does not support tokenizing.")

    def decode(self, request: DecodeRequest) -> DecodeRequestResult:
        raise NotImplementedError("This client does not support decoding.")
