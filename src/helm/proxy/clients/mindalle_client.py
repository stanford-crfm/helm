from typing import Dict, List, Optional

from PIL import Image
from .mindalle.models import Dalle
import numpy as np

from helm.common.cache import CacheConfig, Cache
from helm.common.file_caches.file_cache import FileCache
from helm.common.gpu_utils import get_torch_device_name
from helm.common.hierarchical_logger import hlog, htrack_block
from helm.common.request import Request, RequestResult, TextToImageRequest, Sequence
from helm.common.tokenization_request import (
    DecodeRequest,
    DecodeRequestResult,
    TokenizationRequest,
    TokenizationRequestResult,
)
from .client import Client, wrap_request_time


class MinDALLEClient(Client):
    """
    Source: https://github.com/kakaobrain/mindall-e
    """

    def __init__(self, cache_config: CacheConfig, file_cache: FileCache):
        self._cache = Cache(cache_config)
        self._file_cache: FileCache = file_cache

        self._model: Optional[Dalle] = None

    def _get_model(self) -> Dalle:
        if self._model is None:
            self._model = Dalle.from_pretrained("minDALL-E/1.3B")
            self._model = self._model.to(get_torch_device_name())
        return self._model

    def make_request(self, request: Request) -> RequestResult:
        if not isinstance(request, TextToImageRequest):
            raise ValueError(f"Wrong type of request: {request}")

        raw_request = {
            "prompt": request.prompt,
            # Setting this to a higher value can cause CUDA OOM
            # Fix it to 1 and generate an image `request.num_completions` times
            "num_candidates": 1,
            "softmax_temperature": 1.0,
            "top_k": 256,  # It is recommended that top_k is set lower than 256.
            "top_p": None,
            "device": get_torch_device_name(),
        }

        try:

            def do_it():
                prompt: str = request.prompt

                with htrack_block(f"Generating images for prompt: {prompt}"):
                    model: Dalle = self._get_model()

                    images: List[Image] = []
                    for _ in range(request.num_completions):
                        output = model.sampling(**raw_request).cpu().numpy()
                        output = np.transpose(output, (0, 2, 3, 1))
                        image = Image.fromarray(np.asarray(output[0] * 255, dtype=np.uint8))
                        images.append(image)

                    assert (
                        len(images) == request.num_completions
                    ), f"Expected {request.num_completions} images, but got {len(images)}"

                    result = {"file_locations": []}
                    for image in images:
                        # Write out the image to a file and save the path
                        file_location: str = self._file_cache.get_unique_file_location()
                        image.save(file_location)
                        hlog(f"Image saved at {file_location}.")
                        result["file_locations"].append(file_location)
                    return result

            # Include the model name and number of completions in the cache key
            cache_key: Dict = Client.make_cache_key(
                {"model": request.model_engine, "n": request.num_completions, **raw_request}, request
            )
            results, cached = self._cache.get(cache_key, wrap_request_time(do_it))
        except RuntimeError as e:
            error: str = f"MinDALLEClient error: {e}"
            return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])

        completions: List[Sequence] = [
            Sequence(text="", logprob=0, tokens=[], file_location=file_location)
            for file_location in results["file_locations"]
        ]
        return RequestResult(
            success=True,
            cached=cached,
            request_time=results["request_time"],
            completions=completions,
            embedding=[],
        )

    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        raise NotImplementedError("This client does not support tokenizing.")

    def decode(self, request: DecodeRequest) -> DecodeRequestResult:
        raise NotImplementedError("This client does not support decoding.")
