from threading import Lock
from typing import Any, Dict, List, Optional

import torch
from dataclasses import dataclass
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

from helm.common.cache import CacheConfig
from helm.common.images_utils import open_image
from helm.common.gpu_utils import get_torch_device_name
from helm.common.hierarchical_logger import hlog, htrack_block
from helm.common.media_object import TEXT_TYPE
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.common.request import Request, RequestResult, GeneratedOutput, Token
from helm.common.tokenization_request import TokenizationRequest
from helm.common.request import wrap_request_time
from helm.clients.client import CachingClient, generate_uid_for_multimodal_prompt
from helm.tokenizers.tokenizer import Tokenizer

try:
    from PIL import Image
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, ["images"])

# Added to solve: cutlassF: no kernel found to launch!
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)


@dataclass(frozen=True)
class LoadedPaliGemmaForConditionalGeneration:
    """Loaded model and processor for PaliGemma."""

    model: PaliGemmaForConditionalGeneration
    processor: AutoProcessor


_models_lock: Lock = Lock()
_models: Dict[str, Optional[LoadedPaliGemmaForConditionalGeneration]] = {}


class PaliGemmaClient(CachingClient):
    """
    PaliGemma is a versatile and lightweight vision-language model (VLM) inspired by PaLI-3
    and based on open components such as the SigLIP vision model and the Gemma language model.
    It takes both image and text as input and generates text as output, supporting multiple languages.
    It is designed for class-leading fine-tune performance on a wide range of vision-language tasks
    such as image and short video caption, visual question answering, text reading, object detection
    and object segmentation.
    """

    def __init__(self, tokenizer: Tokenizer, tokenizer_name: str, cache_config: CacheConfig):
        super().__init__(cache_config=cache_config)
        self.tokenizer = tokenizer
        self.tokenizer_name = tokenizer_name
        self._device: str = get_torch_device_name()

    def _get_model(self, checkpoint: str) -> LoadedPaliGemmaForConditionalGeneration:
        global _models_lock
        global _models

        # Ensure that only one thread is loading the model at a time
        with _models_lock:
            if checkpoint not in _models or _models[checkpoint] is None:
                hlog(f"Loading model {checkpoint} and caching in memory...")
                model = PaliGemmaForConditionalGeneration.from_pretrained(
                    checkpoint, torch_dtype=torch.bfloat16, device_map="auto"
                ).eval()
                processor = AutoProcessor.from_pretrained(checkpoint)
                _models[checkpoint] = LoadedPaliGemmaForConditionalGeneration(model, processor)
            loaded_model_processor = _models[checkpoint]

        assert loaded_model_processor is not None
        return loaded_model_processor

    def make_request(self, request: Request) -> RequestResult:
        assert request.multimodal_prompt is not None, "Multimodal prompt is required"

        loaded_model_processor: LoadedPaliGemmaForConditionalGeneration = self._get_model(request.model_deployment)
        model = loaded_model_processor.model
        processor = loaded_model_processor.processor
        generation_args = {"max_new_tokens": request.max_tokens}

        images: List[Image.Image] = []
        prompt_pieces: List[str] = []
        for media_object in request.multimodal_prompt.media_objects:
            if media_object.is_type("image") and media_object.location:
                images += [open_image(media_object.location).convert("RGB")]
            elif media_object.is_type(TEXT_TYPE):
                if media_object.text is None:
                    raise ValueError("MediaObject of text type has missing text field value")
                prompt_pieces.append(media_object.text)
            else:
                raise ValueError(f"Unrecognized MediaObject type {media_object.type}")
        prompt_text: str = "\n".join(prompt_pieces)
        model_inputs = processor(text=prompt_text, images=images, return_tensors="pt").to(self._device)
        input_len = model_inputs["input_ids"].shape[-1]

        completions: List[GeneratedOutput] = []
        with htrack_block(f"Generating for prompt: {prompt_text}"):
            try:
                concat_results = []
                for i_completion in range(request.num_completions):

                    def do_it() -> Dict[str, Any]:
                        with torch.inference_mode():
                            generation = model.generate(
                                **model_inputs, max_new_tokens=request.max_tokens, do_sample=False
                            )[0]
                            if not request.echo_prompt:
                                generation = generation[input_len:]
                            decoded = processor.decode(generation, skip_special_tokens=True)
                            return {"output": decoded}

                    # Include the prompt and model name in the cache key
                    cache_key = CachingClient.make_cache_key(
                        raw_request={
                            "n": request.num_completions,
                            "i": i_completion,
                            "model": request.model,
                            "prompt": generate_uid_for_multimodal_prompt(request.multimodal_prompt),
                            **generation_args,
                        },
                        request=request,
                    )
                    result, cached = self.cache.get(cache_key, wrap_request_time(do_it))
                    concat_results.append(result)
            except RuntimeError as model_error:
                return RequestResult(success=False, cached=False, error=str(model_error), completions=[], embedding=[])

            for result in concat_results:
                text = result["output"]
                hlog(f"Generated text: {text}")
                tokenization_result = self.tokenizer.tokenize(
                    TokenizationRequest(text, tokenizer=self.tokenizer_name, encode=False)
                )
                tokens: List[Token] = [Token(text=str(text), logprob=0) for text in tokenization_result.raw_tokens]
                completions.append(GeneratedOutput(text=text, logprob=0, tokens=tokens))

        return RequestResult(
            success=True,
            cached=cached,
            request_time=result["request_time"],
            completions=completions,
            embedding=[],
        )
