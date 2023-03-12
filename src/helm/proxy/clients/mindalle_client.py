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

import numpy as np
from .mindalle.models import Dalle
# from .mindalle.utils.utils import set_seed


class MinDALLEClient(Client):
    def __init__(self, cache_config: CacheConfig, file_cache: FileCache):
        self._cache = Cache(cache_config)
        self._file_cache: FileCache = file_cache

        self._promptist_model = None
        self._promptist_tokenizer = None

    def _get_model(self) -> Dalle:
        model = Dalle.from_pretrained('minDALL-E/1.3B')
        model = model.to(get_torch_device_name())
        return model

    def make_request(self, request: Request) -> RequestResult:
        if not isinstance(request, TextToImageRequest):
            raise ValueError(f"Wrong type of request: {request}")

        raw_request = {
            "prompt": request.prompt,
            # Setting this to a higher value can cause CUDA OOM
            # Fix it to 1 and generate an image `request.num_completions` times
            "num_candidates": 1,
            "softmax_temperature": 1.0,
            "top_k": 256, # It is recommended that top_k is set lower than 256.
            "top_p": None,
            "device": get_torch_device_name(),
        }

        try:

            def replace_prompt(request_to_update: Dict, new_prompt: str) -> Dict:
                new_request: Dict = dict(request_to_update)
                assert "prompt" in new_request
                new_request["prompt"] = new_prompt
                return new_request

            def do_it():
                prompt: str = request.prompt

                with htrack_block(f"Generating images for prompt: {prompt}"):
                    model: Dalle = self._get_model()
                    promptist_prompt: Optional[str] = None

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
            error: str = f"HuggingFaceDiffusersClient error: {e}"
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
