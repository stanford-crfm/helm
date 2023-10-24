from typing import Dict, List, Optional

from diffusers import DiffusionPipeline
from diffusers.pipelines.stable_diffusion_safe import SafetyConfig
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from helm.common.cache import CacheConfig, Cache
from helm.common.file_caches.file_cache import FileCache
from helm.common.gpu_utils import get_torch_device_name, is_cuda_available
from helm.common.hierarchical_logger import hlog, htrack_block
from helm.common.request import Request, RequestResult, TextToImageRequest, Sequence
from helm.common.tokenization_request import (
    DecodeRequest,
    DecodeRequestResult,
    TokenizationRequest,
    TokenizationRequestResult,
)
from helm.proxy.clients.client import Client, wrap_request_time


class HuggingFaceDiffusersClient(Client):
    def __init__(self, hf_auth_token: str, cache_config: CacheConfig, file_cache: FileCache):
        self._hf_auth_token: str = hf_auth_token
        self._cache = Cache(cache_config)
        self._file_cache: FileCache = file_cache

        self._model_engine_to_diffuser: Dict[str, DiffusionPipeline] = {}
        self._promptist_model = None
        self._promptist_tokenizer = None

    def _get_diffuser(self, model_engine: str) -> DiffusionPipeline:
        """
        Initialize the Diffusion Pipeline based on the model name.
        Cache the model, so it doesn't get reinitialize for a new request.
        """
        if model_engine not in self._model_engine_to_diffuser:
            huggingface_model_name: str
            if model_engine in ["stable-diffusion-v1-4", "promptist-stable-diffusion-v1-4"]:
                huggingface_model_name = "CompVis/stable-diffusion-v1-4"
            elif model_engine == "stable-diffusion-v1-5":
                huggingface_model_name = "runwayml/stable-diffusion-v1-5"
            elif model_engine == "stable-diffusion-v2-base":
                huggingface_model_name = "stabilityai/stable-diffusion-2-base"
            elif model_engine == "stable-diffusion-v2-1-base":
                huggingface_model_name = "stabilityai/stable-diffusion-2-1-base"
            elif model_engine == "dreamlike-diffusion-v1-0":
                huggingface_model_name = "dreamlike-art/dreamlike-diffusion-1.0"
            elif model_engine == "dreamlike-photoreal-v2-0":
                huggingface_model_name = "dreamlike-art/dreamlike-photoreal-2.0"
            elif model_engine == "openjourney-v1-0":
                huggingface_model_name = "prompthero/openjourney"
            elif model_engine == "openjourney-v2-0":
                huggingface_model_name = "prompthero/openjourney-v2"
            elif model_engine == "redshift-diffusion":
                huggingface_model_name = "nitrosocke/redshift-diffusion"
            elif "stable-diffusion-safe" in model_engine:
                huggingface_model_name = "AIML-TUDA/stable-diffusion-safe"
            elif model_engine == "vintedois-diffusion-v0-1":
                huggingface_model_name = "22h/vintedois-diffusion-v0-1"
            else:
                raise ValueError(f"Unhandled model: {model_engine}")

            pipeline = DiffusionPipeline.from_pretrained(
                huggingface_model_name,
                torch_dtype=torch.float16 if is_cuda_available() else torch.float,
                use_auth_token=self._hf_auth_token,
            )
            self._model_engine_to_diffuser[model_engine] = pipeline.to(get_torch_device_name())
        return self._model_engine_to_diffuser[model_engine]

    def make_request(self, request: Request) -> RequestResult:
        if not isinstance(request, TextToImageRequest):
            raise ValueError(f"Wrong type of request: {request}")

        raw_request = {
            "prompt": request.prompt,
            # Setting this to a higher value can cause CUDA OOM
            # Fix it to 1 and generate an image `request.num_completions` times
            "num_images_per_prompt": 1,
        }
        if request.guidance_scale is not None:
            raw_request["guidance_scale"] = request.guidance_scale
        if request.steps is not None:
            raw_request["num_inference_steps"] = request.steps
        if request.width is not None:
            raw_request["width"] = request.width
        if request.height is not None:
            raw_request["height"] = request.height

        # Add the additional pre-configured parameters for Safe Stable Diffusion
        if request.model_engine == "stable-diffusion-safe-weak":
            raw_request = {**raw_request, **SafetyConfig.WEAK}
        elif request.model_engine == "stable-diffusion-safe-medium":
            raw_request = {**raw_request, **SafetyConfig.MEDIUM}
        elif request.model_engine == "stable-diffusion-safe-strong":
            raw_request = {**raw_request, **SafetyConfig.STRONG}
        elif request.model_engine == "stable-diffusion-safe-max":
            raw_request = {**raw_request, **SafetyConfig.MAX}

        try:

            def replace_prompt(request_to_update: Dict, new_prompt: str) -> Dict:
                new_request: Dict = dict(request_to_update)
                assert "prompt" in new_request
                new_request["prompt"] = new_prompt
                return new_request

            def do_it():
                prompt: str = request.prompt

                with htrack_block(f"Generating images for prompt: {prompt}"):
                    diffuser: DiffusionPipeline = self._get_diffuser(request.model_engine)
                    promptist_prompt: Optional[str] = None

                    images: List[Image] = []
                    for _ in range(request.num_completions):
                        if request.model_engine == "promptist-stable-diffusion-v1-4":
                            promptist_prompt = self._generate_promptist_version(prompt)
                            hlog(f"Promptist: {prompt} -> {promptist_prompt}")
                            image = diffuser(**replace_prompt(raw_request, promptist_prompt)).images[0]
                        elif request.model_engine == "openjourney-v1-0":
                            # It is required to include "mdjrny-v4 style" in prompt for Openjourney v1
                            image = diffuser(**replace_prompt(raw_request, f"mdjrny-v4 style {prompt}")).images[0]
                        elif request.model_engine == "redshift-diffusion":
                            # It is required to include "redshift style" to generate 3D images
                            image = diffuser(**replace_prompt(raw_request, f"redshift style {prompt}")).images[0]
                        else:
                            image = diffuser(**raw_request).images[0]
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
