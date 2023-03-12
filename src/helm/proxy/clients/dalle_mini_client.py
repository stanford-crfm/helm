from typing import Dict, List, Optional

from diffusers import DiffusionPipeline
from diffusers.pipelines.stable_diffusion_safe import SafetyConfig
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

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

#For DALLE mini
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
from tqdm import tqdm
from functools import partial
from transformers import CLIPProcessor, FlaxCLIPModel
from flax.jax_utils import replicate
from flax.training.common_utils import shard_prng_key
from .dalle_mini.vqgan_jax.modeling_flax_vqgan import VQModel
from .dalle_mini import DalleBart, DalleBartProcessor

VQGAN_REPO = "dalle-mini/vqgan_imagenet_f16_16384"
VQGAN_COMMIT_ID = "e93a26e7707683d349bf5d5c41c5b0ef69b677a9"


class DALLEMiniClient(Client):
    def __init__(self, cache_config: CacheConfig, file_cache: FileCache):
        self._cache = Cache(cache_config)
        self._file_cache: FileCache = file_cache

        self._model_engine_to_model = {}
        self._promptist_model = None
        self._promptist_tokenizer = None

    def _get_model(self, model_engine: str):
        """
        Initialize the model based on the model name.
        Cache the model, so it doesn't get reinitialize for a new request.
        """
        if model_engine not in self._model_engine_to_model:
            model_name: str
            if model_engine == "dalle-mini":
                model_name = "dalle-mini/dalle-mini/mini-1:v0"
            elif model_engine == "dalle-mega":
                model_name = "dalle-mini/dalle-mini/mega-1-fp16:latest"
            else:
                raise ValueError(f"Unhandled model: {model_engine}")

            model, params = DalleBart.from_pretrained(
                model_name, revision=None, dtype=jnp.float16, _do_init=False
            )
            processor = DalleBartProcessor.from_pretrained(
                model_name, revision=None
            )
            vqgan, vqgan_params = VQModel.from_pretrained(
                VQGAN_REPO, revision=VQGAN_COMMIT_ID, _do_init=False
            )
            params = replicate(params)
            vqgan_params = replicate(vqgan_params)
            self._model_engine_to_model[model_engine] = [model, params, processor, vqgan, vqgan_params]
        return self._model_engine_to_model[model_engine]

    def make_request(self, request: Request) -> RequestResult:
        if not isinstance(request, TextToImageRequest):
            raise ValueError(f"Wrong type of request: {request}")

        raw_request = {
            "prompt": request.prompt,
            "top_k": None,
            "top_p": None,
            "temperature": None,
            "condition_scale": 10.0,
        }

        try:

            def replace_prompt(request_to_update: Dict, new_prompt: str) -> Dict:
                new_request: Dict = dict(request_to_update)
                assert "prompt" in new_request
                new_request["prompt"] = new_prompt
                return new_request

            def _inference(
                model, params, vqgan, vqgan_params,
                tokenized_prompt, subkey,
                top_k, top_p, temperature, condition_scale
            ):
                @partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(3, 4, 5, 6))
                def p_generate(
                    tokenized_prompt, key, params, top_k, top_p, temperature, condition_scale
                ):
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

            def do_it():
                prompt: str = request.prompt

                with htrack_block(f"Generating images for prompt: {prompt}"):
                    model, params, processor, vqgan, vqgan_params = self._get_model(request.model_engine)
                    tokenized_prompts = processor([prompt])
                    tokenized_prompt = replicate(tokenized_prompts)

                    promptist_prompt: Optional[str] = None

                    images: List[Image] = []
                    key = jax.random.PRNGKey(0)
                    for _ in range(request.num_completions):
                        key, subkey = jax.random.split(key)
                        image = _inference(
                            model, params, vqgan, vqgan_params,
                            tokenized_prompt, subkey,
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
                    if promptist_prompt is not None:
                        # Save the Promptist version of the prompts in the cache, just in case we need it later
                        result["promptist_prompt"] = promptist_prompt

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
            error: str = f"DALLEMiniClient error: {e}"
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

    def _generate_promptist_version(self, prompt: str) -> str:
        """
        Generate a better version of the prompt with Promptist.
        Promptist was trained specifically with CompVis/stable-diffusion-v1-4.
        Adapted from https://huggingface.co/spaces/microsoft/Promptist/blob/main/app.py.
        """

        def load_promptist():
            prompter_model = AutoModelForCausalLM.from_pretrained("microsoft/Promptist")
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"
            return prompter_model, tokenizer

        def generate(plain_text: str) -> str:
            if self._promptist_model is None or self._promptist_tokenizer is None:
                self._promptist_model, self._promptist_tokenizer = load_promptist()
            assert self._promptist_model is not None
            assert self._promptist_tokenizer is not None

            input_ids = self._promptist_tokenizer(f"{plain_text.strip()} Rephrase:", return_tensors="pt").input_ids
            eos_id = self._promptist_tokenizer.eos_token_id
            # Used the same hyperparameters from the example
            outputs = self._promptist_model.generate(
                input_ids,
                do_sample=False,
                max_new_tokens=75,
                num_beams=8,
                num_return_sequences=8,
                eos_token_id=eos_id,
                pad_token_id=eos_id,
                length_penalty=-1.0,
            )
            output_texts: List[str] = self._promptist_tokenizer.batch_decode(outputs, skip_special_tokens=True)

            for output_text in output_texts:
                res: str = output_text.replace(f"{plain_text} Rephrase:", "").strip()
                # The Promptist model sometimes generates empty string results.
                # Return the first non-empty string result.
                if len(res) > 0:
                    return res

            # If all fails, just return the original text.
            return plain_text

        return generate(prompt)

    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        raise NotImplementedError("This client does not support tokenizing.")

    def decode(self, request: DecodeRequest) -> DecodeRequestResult:
        raise NotImplementedError("This client does not support decoding.")
