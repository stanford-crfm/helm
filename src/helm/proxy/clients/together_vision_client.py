from typing import List, Dict, Optional
import base64
import requests

from helm.common.cache import CacheConfig
from helm.common.file_cache import FileCache
from helm.common.request import Request, RequestResult, Sequence, TextToImageRequest
from helm.common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
    DecodeRequest,
    DecodeRequestResult,
)

from .client import Client, wrap_request_time
from .together_client import TogetherClient


class TogetherVisionClient(TogetherClient):
    """
    Client for image generation via the Together API.
    """

    DEFAULT_IMAGE_HEIGHT: int = 512
    DEFAULT_IMAGE_WIDTH: int = 512

    def __init__(self, cache_config: CacheConfig, file_cache_path: str, api_key: Optional[str] = None):
        super().__init__(cache_config, api_key)
        self.file_cache: FileCache = FileCache(file_cache_path, "png")

    def make_request(self, request: Request) -> RequestResult:
        assert isinstance(request, TextToImageRequest)
        # Following https://docs.together.xyz/en/api
        # TODO: are there other parameters for StableDiffusion?
        # TODO: support the four different SafeStableDiffusion configurations
        raw_request = {
            "request_type": "image-model-inference",
            "model": request.model_engine,
            "prompt": request.prompt,
            "n": request.num_completions,
            "guidance_scale": request.guidance_scale,
        }
        if request.width and request.height:
            raw_request["width"] = request.width
            raw_request["height"] = request.height
        else:
            raw_request["width"] = self.DEFAULT_IMAGE_WIDTH
            raw_request["height"] = self.DEFAULT_IMAGE_HEIGHT

        cache_key: Dict = Client.make_cache_key(raw_request, request)

        try:

            def do_it():
                result = requests.post(self.INFERENCE_ENDPOINT, json=raw_request).json()
                assert "output" in result, f"Invalid response: {result}"

                for choice in result["output"]["choices"]:
                    # Write out the image to a file and save the path
                    file_path: str = self.file_cache.store(lambda: base64.b64decode(choice["image_base64"]))
                    choice["file_path"] = file_path
                return result["output"]

            response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except RuntimeError as e:
            error: str = f"TogetherVisionClient error: {e}"
            return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])

        completions: List[Sequence] = [
            Sequence(text="", logprob=0, tokens=[], file_path=choice["file_path"]) for choice in response["choices"]
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
