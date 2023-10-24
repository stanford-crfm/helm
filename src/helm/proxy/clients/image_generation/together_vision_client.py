from typing import List, Dict, Optional
import base64
import requests

from helm.common.cache import CacheConfig
from helm.common.file_caches.file_cache import FileCache
from helm.common.request import Request, RequestResult, Sequence, TextToImageRequest
from helm.common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
    DecodeRequest,
    DecodeRequestResult,
)

from helm.proxy.clients.client import Client, wrap_request_time
from helm.proxy.clients.together_client import TogetherClient


class TogetherVisionClient(TogetherClient):
    """
    Client for image generation via the Together API.
    """

    DEFAULT_IMAGE_HEIGHT: int = 512
    DEFAULT_IMAGE_WIDTH: int = 512

    DEFAULT_GUIDANCE_SCALE: float = 7.5
    DEFAULT_STEPS: int = 50

    def __init__(self, cache_config: CacheConfig, file_cache: FileCache, api_key: Optional[str] = None):
        super().__init__(cache_config, api_key)
        self.file_cache: FileCache = file_cache

        self._promptist_model = None
        self._promptist_tokenizer = None

    def make_request(self, request: Request) -> RequestResult:
        if not isinstance(request, TextToImageRequest):
            raise ValueError(f"Wrong type of request: {request}")

        # Following https://docs.together.xyz/en/api
        raw_request = {
            "request_type": "image-model-inference",
            "model": request.model_engine,
            "prompt": request.prompt,
            "n": request.num_completions,
            "guidance_scale": request.guidance_scale
            if request.guidance_scale is not None
            else self.DEFAULT_GUIDANCE_SCALE,
            "steps": request.steps if request.steps is not None else self.DEFAULT_STEPS,
        }

        if request.width is None or request.height is None:
            raw_request["width"] = self.DEFAULT_IMAGE_WIDTH
            raw_request["height"] = self.DEFAULT_IMAGE_HEIGHT
        else:
            raw_request["width"] = request.width
            raw_request["height"] = request.height

        cache_key: Dict = Client.make_cache_key(raw_request, request)

        try:

            def do_it():
                result = requests.post(self.INFERENCE_ENDPOINT, json=raw_request).json()
                assert "output" in result, f"Invalid response: {result} from prompt: {request.prompt}"

                for choice in result["output"]["choices"]:
                    # Write out the image to a file and save the path
                    choice["file_path"] = self.file_cache.store(lambda: base64.b64decode(choice["image_base64"]))
                    choice.pop("image_base64", None)
                return result["output"]

            response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except RuntimeError as e:
            error: str = f"TogetherVisionClient error: {e}"
            return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])

        completions: List[Sequence] = [
            Sequence(text="", logprob=0, tokens=[], file_location=choice["file_path"]) for choice in response["choices"]
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
