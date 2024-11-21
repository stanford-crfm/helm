from typing import Any, Dict, List, Optional
import base64
import requests

from helm.common.cache import CacheConfig, Cache
from helm.common.file_caches.file_cache import FileCache
from helm.common.request import Request, RequestResult, GeneratedOutput, wrap_request_time
from helm.common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
    DecodeRequest,
    DecodeRequestResult,
)

from helm.clients.client import CachingClient, Client
from helm.clients.image_generation.image_generation_client_utils import get_single_image_multimedia_object


class TogetherImageGenerationClient(Client):
    """
    Client for image generation via the Together API.
    """

    DEFAULT_IMAGE_HEIGHT: int = 512
    DEFAULT_IMAGE_WIDTH: int = 512

    DEFAULT_GUIDANCE_SCALE: float = 7.5
    DEFAULT_STEPS: int = 50

    INFERENCE_ENDPOINT: str = "https://api.together.xyz/api/inference"

    def __init__(self, cache_config: CacheConfig, file_cache: FileCache, api_key: Optional[str] = None):
        self._cache = Cache(cache_config)
        self.file_cache: FileCache = file_cache

        self._promptist_model = None
        self._promptist_tokenizer = None

        self.api_key: Optional[str] = api_key

    def make_request(self, request: Request) -> RequestResult:
        # Following https://docs.together.xyz/en/api
        assert request.image_generation_parameters is not None
        raw_request = {
            "request_type": "image-model-inference",
            "model": request.model_engine,
            "prompt": request.prompt,
            "n": request.num_completions,
            "guidance_scale": (
                request.image_generation_parameters.guidance_scale
                if request.image_generation_parameters.guidance_scale is not None
                else self.DEFAULT_GUIDANCE_SCALE
            ),
            "steps": (
                request.image_generation_parameters.diffusion_denoising_steps
                if request.image_generation_parameters.diffusion_denoising_steps is not None
                else self.DEFAULT_STEPS
            ),
        }

        if (
            request.image_generation_parameters.output_image_width is None
            or request.image_generation_parameters.output_image_height is None
        ):
            raw_request["width"] = self.DEFAULT_IMAGE_WIDTH
            raw_request["height"] = self.DEFAULT_IMAGE_HEIGHT
        else:
            raw_request["width"] = request.image_generation_parameters.output_image_width
            raw_request["height"] = request.image_generation_parameters.output_image_height

        cache_key = CachingClient.make_cache_key(raw_request, request)

        try:

            def do_it() -> Dict[str, Any]:
                result = requests.post(self.INFERENCE_ENDPOINT, json=raw_request).json()
                assert "output" in result, f"Invalid response: {result} from prompt: {request.prompt}"

                for choice in result["output"]["choices"]:
                    # Write out the image to a file and save the path
                    choice["file_path"] = self.file_cache.store(lambda: base64.b64decode(choice["image_base64"]))
                    choice.pop("image_base64", None)
                return result["output"]

            response, cached = self._cache.get(cache_key, wrap_request_time(do_it))
        except RuntimeError as e:
            error: str = f"TogetherVisionClient error: {e}"
            return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])

        completions: List[GeneratedOutput] = [
            GeneratedOutput(
                text="",
                logprob=0,
                tokens=[],
                multimodal_content=get_single_image_multimedia_object(choice["file_path"]),
            )
            for choice in response["choices"]
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
