from threading import Lock
from typing import List, Dict, Tuple, Optional

import torch

from helm.common.cache import Cache, CacheConfig
from helm.common.file_caches.file_cache import FileCache
from helm.common.gpu_utils import is_cuda_available
from helm.common.hierarchical_logger import hlog, htrack_block
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.common.request import Request, RequestResult, Sequence
from helm.common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
    DecodeRequest,
    DecodeRequestResult,
)
from helm.proxy.clients.client import Client, CachingClient
from .image_generation_client_utils import get_single_image_multimedia_object


_models_lock: Lock = Lock()
_models: Dict[str, Tuple] = {}


class DeepFloydClient(Client):
    """
    Client for [DeepFloyd image generation models](https://huggingface.co/DeepFloyd/IF-I-M-v1.0).
    DeepFloyd-IF is a pixel-based text-to-image triple-cascaded diffusion model, that can generate pictures with
    new state-of-the-art for photorealism and language understanding.
    """

    MODEL_NAME_TO_MODELS: Dict[str, Tuple[str, str]] = {
        "IF-I-XL-v1.0": ("DeepFloyd/IF-I-XL-v1.0", "DeepFloyd/IF-II-L-v1.0"),  # XL
        "IF-I-L-v1.0": ("DeepFloyd/IF-I-L-v1.0", "DeepFloyd/IF-II-L-v1.0"),  # Large
        "IF-I-M-v1.0": ("DeepFloyd/IF-I-M-v1.0", "DeepFloyd/IF-II-M-v1.0"),  # Medium
    }

    @staticmethod
    def initialize_model(stage1_model_name: str, stage2_model_name: str, auth_token: Optional[str]):
        try:
            from diffusers import DiffusionPipeline
        except ModuleNotFoundError as e:
            handle_module_not_found_error(e, ["heim"])

        with htrack_block(f"Initializing the three stages of the IF model: {stage1_model_name}"):
            cuda_available: bool = is_cuda_available()
            torch_dtype: torch.dtype = torch.float16 if cuda_available else torch.float

            # stage 1
            stage_1 = DiffusionPipeline.from_pretrained(
                stage1_model_name, torch_dtype=torch_dtype, use_auth_token=auth_token
            )
            if cuda_available:
                stage_1.enable_model_cpu_offload()

            # stage 2
            stage_2 = DiffusionPipeline.from_pretrained(
                stage2_model_name, text_encoder=None, torch_dtype=torch_dtype, use_auth_token=auth_token
            )
            if cuda_available:
                stage_2.enable_model_cpu_offload()

            # stage 3
            safety_modules = {
                "feature_extractor": stage_1.feature_extractor,
                "safety_checker": stage_1.safety_checker,
                "watermarker": stage_1.watermarker,
            }
            stage_3 = DiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-x4-upscaler",
                **safety_modules,
                torch_dtype=torch_dtype,
                use_auth_token=auth_token,
            )
            if cuda_available:
                stage_3.enable_model_cpu_offload()

            return stage_1, stage_2, stage_3

    @staticmethod
    def convert_to_raw_request(request: Request) -> Dict:
        # Use default hyperparameters for everything else
        raw_request: Dict = {
            "model": request.model_engine,
            "n": request.num_completions,
            "prompt": request.prompt,
            "request_type": "image-model-inference",
        }
        if request.random is not None:
            raw_request["random"] = request.random
        return raw_request

    def _get_model(self, model_name: str) -> Tuple:
        global _models_lock
        global _models

        with _models_lock:
            if model_name not in _models:
                stage1_model_name, stage2_model_name = self.MODEL_NAME_TO_MODELS[model_name]
                _models[model_name] = self.initialize_model(
                    stage1_model_name, stage2_model_name, auth_token=self._hf_auth_token
                )
            return _models[model_name]

    def __init__(self, hf_auth_token: str, cache_config: CacheConfig, file_cache: FileCache):
        self._hf_auth_token: Optional[str] = hf_auth_token
        self._cache = Cache(cache_config)
        self._file_cache: FileCache = file_cache

    def make_request(self, request: Request) -> RequestResult:
        if request.model_engine not in self.MODEL_NAME_TO_MODELS:
            raise ValueError(f"Unsupported model: {request.model_engine}")

        raw_request = DeepFloydClient.convert_to_raw_request(request)
        cache_key: Dict = CachingClient.make_cache_key(raw_request, request)

        stage_1, stage_2, stage_3 = self._get_model(request.model_engine)

        try:

            def do_it():
                prompt: str = request.prompt
                with htrack_block(f"Generating images for prompt: {prompt}"):
                    image_paths: List[str] = []

                    for i in range(request.num_completions):
                        prompt_embeds, negative_embeds = stage_1.encode_prompt(prompt)

                        generator = torch.manual_seed(i)
                        image = stage_1(
                            prompt_embeds=prompt_embeds,
                            negative_prompt_embeds=negative_embeds,
                            generator=generator,
                            output_type="pt",
                        ).images

                        image = stage_2(
                            image=image,
                            prompt_embeds=prompt_embeds,
                            negative_prompt_embeds=negative_embeds,
                            generator=generator,
                            output_type="pt",
                        ).images

                        image = stage_3(prompt=prompt, image=image, generator=generator, noise_level=100).images

                        file_location: str = self._file_cache.generate_unique_new_file_path()  # type: ignore
                        image[0].save(file_location)
                        hlog(f"Image saved at {file_location}")
                        image_paths.append(file_location)

                    return {"images": image_paths}

            response, cached = self._cache.get(cache_key, do_it)
        except RuntimeError as e:
            error: str = f"DeepFloyd Client error: {e}"
            return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])

        completions: List[Sequence] = [
            Sequence(text="", logprob=0, tokens=[], multimodal_content=get_single_image_multimedia_object(file_path))
            for file_path in response["images"]
        ]
        # To maintain backwards compatibility with the deepfloyd script
        request_time: float = (
            response["total_inference_time"] if "total_inference_time" in response else response["request_time"]
        )
        return RequestResult(
            success=True,
            cached=cached,
            request_time=request_time,
            completions=completions,
            embedding=[],
        )

    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        raise NotImplementedError("This client does not support tokenizing.")

    def decode(self, request: DecodeRequest) -> DecodeRequestResult:
        raise NotImplementedError("This client does not support decoding.")
