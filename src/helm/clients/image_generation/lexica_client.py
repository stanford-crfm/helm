from typing import Any, List, Dict, Union
import base64
import requests
import urllib.parse

from helm.common.cache import CacheConfig, Cache
from helm.common.file_caches.file_cache import FileCache
from helm.common.images_utils import encode_base64
from helm.common.request import Request, RequestResult, GeneratedOutput, wrap_request_time
from helm.common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
    DecodeRequest,
    DecodeRequestResult,
)
from helm.clients.client import Client, CachingClient
from helm.clients.image_generation.image_generation_client_utils import get_single_image_multimedia_object


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
        if request.model_engine != "search-stable-diffusion-1.5":
            # Only Stable Diffusion 1.5 is supported at the moment
            raise ValueError(f"Invalid model: {request.model_engine}")

        raw_request: Dict[str, Union[str, int]] = {
            "model": request.model_engine,
            "prompt": request.prompt,
            "n": request.num_completions,
        }
        cache_key = CachingClient.make_cache_key(raw_request, request)

        try:

            def do_it() -> Dict[str, Any]:
                num_completions: int = int(raw_request["n"])
                result = requests.get(
                    f"https://lexica.art/api/v1/search?{urllib.parse.urlencode({'q': request.prompt})}"
                ).json()
                assert "images" in result, f"Invalid response: {result} from prompt: {request.prompt}"
                assert len(result["images"]) >= num_completions, "Did not retrieve enough images"

                image_locations: List[str] = []
                # Most relevant images are at the top of the list
                for image in result["images"][:num_completions]:
                    # Write out the image to a file and save the location
                    image_base64: str = encode_base64(image["src"])
                    image_locations.append(self.file_cache.store(lambda: base64.b64decode(image_base64)))
                return {"image_locations": image_locations}

            response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except RuntimeError as e:
            error: str = f"LexicaClient error: {e}"
            return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])

        completions: List[GeneratedOutput] = [
            GeneratedOutput(
                text="", logprob=0, tokens=[], multimodal_content=get_single_image_multimedia_object(location)
            )
            for location in response["image_locations"]
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
