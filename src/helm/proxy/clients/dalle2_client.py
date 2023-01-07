from typing import Any, Dict, List, Optional
import base64

import openai

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
from .openai_client import OpenAIClient


class DALLE2Client(OpenAIClient):
    MAX_PROMPT_LENGTH: int = 1000
    VALID_IMAGE_DIMENSIONS: List[int] = [256, 512, 1024]
    DEFAULT_IMAGE_SIZE_STR: str = "512x512"

    def __init__(
        self,
        api_key: str,
        cache_config: CacheConfig,
        file_cache_path: str,
        org_id: Optional[str] = None,
    ):
        super().__init__(api_key, cache_config, org_id=org_id)
        self.file_cache: FileCache = FileCache(file_cache_path, "png")

    def make_request(self, request: Request) -> RequestResult:
        def get_size_str(w: Optional[int], h: Optional[int]) -> str:
            if w is None or h is None:
                return self.DEFAULT_IMAGE_SIZE_STR

            assert w == h, "The DALL-E 2 API only supports generating square images."
            assert w in self.VALID_IMAGE_DIMENSIONS, "Valid dimensions are 256x256, 512x512, or 1024x1024 pixels."
            return f"{w}x{h}"

        assert isinstance(request, TextToImageRequest)
        assert len(request.prompt) <= self.MAX_PROMPT_LENGTH, "The maximum length of the prompt is 1000 characters."
        assert 1 <= request.num_completions <= 10, "`num_completions` must be between 1 and 10."

        # https://beta.openai.com/docs/api-reference/images/create#images/create-response_format
        raw_request: Dict[str, Any] = {
            "prompt": request.prompt,
            "n": request.num_completions,
            "size": get_size_str(request.width, request.height),
            "response_format": "b64_json",  # Always set to b64_json as URLs are only valid for an hour
        }

        try:

            def do_it():
                openai.organization = self.org_id
                openai.api_key = self.api_key
                openai.api_base = self.api_base
                result = openai.Image.create(**raw_request)
                assert "data" in result, f"Invalid response: {result}"

                for image in result["data"]:
                    # Write out the image to a file and save the path
                    image["file_path"] = self.file_cache.store(lambda: base64.b64decode(image["b64_json"]))
                return result

            cache_key = Client.make_cache_key(raw_request, request)
            response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except openai.error.OpenAIError as e:
            error: str = f"DALL-E 2 error: {e}"
            return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])

        completions: List[Sequence] = [
            Sequence(text="", logprob=0, tokens=[], file_path=generated_image["file_path"])
            for generated_image in response["data"]
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
