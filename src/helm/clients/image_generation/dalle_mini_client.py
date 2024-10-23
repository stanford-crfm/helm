from typing import Any, Dict, List

import numpy as np
from functools import partial

from helm.common.cache import CacheConfig, Cache
from helm.common.file_caches.file_cache import FileCache
from helm.common.hierarchical_logger import hlog, htrack_block
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.common.request import Request, RequestResult, GeneratedOutput, wrap_request_time
from helm.common.tokenization_request import (
    DecodeRequest,
    DecodeRequestResult,
    TokenizationRequest,
    TokenizationRequestResult,
)
from helm.clients.client import Client, CachingClient
from helm.clients.image_generation.image_generation_client_utils import get_single_image_multimedia_object


class DALLEMiniClient(Client):
    """
    Source: https://github.com/borisdayma/dalle-mini, https://github.com/patil-suraj/vqgan-jax
    """

    VQGAN_REPO = "dalle-mini/vqgan_imagenet_f16_16384"
    VQGAN_COMMIT_ID = "e93a26e7707683d349bf5d5c41c5b0ef69b677a9"

    def __init__(self, cache_config: CacheConfig, file_cache: FileCache):
        self._cache = Cache(cache_config)
        self._file_cache: FileCache = file_cache

        self._model_engine_to_model = {}

    def _get_model(self, model_engine: str):
        """
        Initialize the model based on the model name.
        Cache the model, so it doesn't get reinitialize for a new request.
        """
        try:
            import jax.numpy as jnp
            from flax.jax_utils import replicate

            from helm.clients.image_generation.dalle_mini.vqgan_jax.modeling_flax_vqgan import VQModel
            from helm.clients.image_generation.dalle_mini import DalleBart, DalleBartProcessor
        except ModuleNotFoundError as e:
            handle_module_not_found_error(e, ["heim"])

        if model_engine not in self._model_engine_to_model:
            model_name: str
            if model_engine == "dalle-mini":
                model_name = "dalle-mini/dalle-mini/mini-1:v0"
            elif model_engine == "dalle-mega":
                model_name = "dalle-mini/dalle-mini/mega-1-fp16:latest"
            else:
                raise ValueError(f"Unhandled model: {model_engine}")

            model, params = DalleBart.from_pretrained(model_name, revision=None, dtype=jnp.float16, _do_init=False)
            processor = DalleBartProcessor.from_pretrained(model_name, revision=None)
            vqgan, vqgan_params = VQModel.from_pretrained(
                self.VQGAN_REPO, revision=self.VQGAN_COMMIT_ID, _do_init=False
            )
            params = replicate(params)
            vqgan_params = replicate(vqgan_params)
            self._model_engine_to_model[model_engine] = [model, params, processor, vqgan, vqgan_params]
        return self._model_engine_to_model[model_engine]

    def make_request(self, request: Request) -> RequestResult:
        try:
            import jax
            from flax.training.common_utils import shard_prng_key
            from flax.jax_utils import replicate
            from PIL import Image
        except ModuleNotFoundError as e:
            handle_module_not_found_error(e, ["heim"])

        raw_request = {
            "prompt": request.prompt,
            "top_k": None,
            "top_p": None,
            "temperature": None,
            "condition_scale": 10.0,
        }

        try:

            def _inference(
                model, params, vqgan, vqgan_params, tokenized_prompt, subkey, top_k, top_p, temperature, condition_scale
            ):
                @partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(3, 4, 5, 6))
                def p_generate(tokenized_prompt, key, params, top_k, top_p, temperature, condition_scale):
                    return model.generate(
                        **tokenized_prompt,
                        prng_key=key,
                        params=params,
                        top_k=top_k,
                        top_p=top_p,
                        temperature=temperature,
                        condition_scale=condition_scale,
                    )

                @partial(jax.pmap, axis_name="batch")
                def p_decode(indices, params):
                    return vqgan.decode_code(indices, params=params)

                # generate images
                encoded_images = p_generate(
                    tokenized_prompt,
                    shard_prng_key(subkey),
                    params,
                    top_k,
                    top_p,
                    temperature,
                    condition_scale,
                )
                # remove BOS
                encoded_images = encoded_images.sequences[..., 1:]
                # decode images
                decoded_images = p_decode(encoded_images, vqgan_params)
                decoded_images = decoded_images.clip(0.0, 1.0).reshape((-1, 256, 256, 3))
                return decoded_images

            def do_it() -> Dict[str, Any]:
                prompt: str = request.prompt

                with htrack_block(f"Generating images for prompt: {prompt}"):
                    model, params, processor, vqgan, vqgan_params = self._get_model(request.model_engine)
                    tokenized_prompts = processor([prompt])
                    tokenized_prompt = replicate(tokenized_prompts)

                    images: List[Image] = []
                    key = jax.random.PRNGKey(0)
                    for _ in range(request.num_completions):
                        key, subkey = jax.random.split(key)
                        image = _inference(
                            model,
                            params,
                            vqgan,
                            vqgan_params,
                            tokenized_prompt,
                            subkey,
                            raw_request["top_k"],
                            raw_request["top_p"],
                            raw_request["temperature"],
                            raw_request["condition_scale"],
                        )[0]
                        image = Image.fromarray(np.asarray(image * 255, dtype=np.uint8))
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
            cache_key = CachingClient.make_cache_key(
                {"model": request.model_engine, "n": request.num_completions, **raw_request}, request
            )
            results, cached = self._cache.get(cache_key, wrap_request_time(do_it))
        except RuntimeError as e:
            error: str = f"DALLEMiniClient error: {e}"
            return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])

        completions: List[GeneratedOutput] = [
            GeneratedOutput(
                text="", logprob=0, tokens=[], multimodal_content=get_single_image_multimedia_object(file_location)
            )
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
