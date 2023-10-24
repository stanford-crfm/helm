from typing import List, Dict
import base64
import requests
import urllib.parse

from helm.common.cache import CacheConfig, Cache
from helm.common.file_caches.file_cache import FileCache
from helm.common.images_utils import encode_base64
from helm.common.request import Request, RequestResult, Sequence, TextToImageRequest
from helm.common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
    DecodeRequest,
    DecodeRequestResult,
)

from helm.proxy.clients.client import Client, wrap_request_time


class LexicaClient(Client):
    """
    Client for Lexica API. Does not support image generation.
    """

    def __init__(self, cache_config: CacheConfig, file_cache: FileCache):
        self.cache = Cache(cache_config)
        self.file_cache: FileCache = file_cache

    def make_request(self, request: Request) -> RequestResult:
        """
        Retrieves images through Lexica's search API (https://lexica.art/docs).
        The search API is powered by CLIP to fetch the most relevant images for a given query.
        """
        if not isinstance(request, TextToImageRequest):
            raise ValueError(f"Wrong type of request: {request}")
        if request.model_engine != "search-stable-diffusion-1.5":
            # Only Stable Diffusion 1.5 is supported at the moment
            raise ValueError(f"Invalid model: {request.model_engine}")

        raw_request = {
            "model": request.model_engine,
            "prompt": request.prompt,
            "n": request.num_completions,
        }
        cache_key: Dict = Client.make_cache_key(raw_request, request)

        try:

            def do_it():
                result = requests.get(
                    f"https://lexica.art/api/v1/search?{urllib.parse.urlencode({'q': request.prompt})}"
                ).json()
                assert "images" in result, f"Invalid response: {result} from prompt: {request.prompt}"
                assert len(result["images"]) >= raw_request["n"], "Did not retrieve enough images"

                image_locations: List[str] = []
                # Most relevant images are at the top of the list
                for image in result["images"][: raw_request["n"]]:
                    # Write out the image to a file and save the location
                    image_base64: str = encode_base64(image["src"])
                    image_locations.append(self.file_cache.store(lambda: base64.b64decode(image_base64)))
                return {"image_locations": image_locations}

            response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except RuntimeError as e:
            error: str = f"LexicaClient error: {e}"
            return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])

        completions: List[Sequence] = [
            Sequence(text="", logprob=0, tokens=[], file_location=image_location)
            for image_location in response["image_locations"]
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
