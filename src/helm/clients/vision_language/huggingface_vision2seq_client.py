from threading import Lock
from typing import Any, Dict, List, Optional

from dataclasses import dataclass
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image
import torch

from helm.common.cache import CacheConfig
from helm.common.gpu_utils import get_torch_device_name, is_cuda_available
from helm.common.hierarchical_logger import hlog, htrack_block
from helm.common.media_object import TEXT_TYPE
from helm.common.request import Request, RequestResult, GeneratedOutput, Token
from helm.common.request import wrap_request_time
from helm.common.tokenization_request import TokenizationRequest
from helm.clients.client import CachingClient, generate_uid_for_multimodal_prompt
from helm.tokenizers.tokenizer import Tokenizer


@dataclass(frozen=True)
class Vision2SeqModelProcessor:
    """Loaded model and processor."""

    model: AutoModelForVision2Seq
    processor: AutoProcessor


_models_lock: Lock = Lock()
_models: Dict[str, Optional[Vision2SeqModelProcessor]] = {
    "HuggingFaceM4/idefics2-8b": None,
}


class HuggingFaceVision2SeqClient(CachingClient):
    """
    Models for Vision2Seq models from HuggingFace.
    """

    ASSISTANT_PREFIX: str = "Assistant:"

    def __init__(self, tokenizer: Tokenizer, tokenizer_name: str, cache_config: CacheConfig):
        super().__init__(cache_config=cache_config)
        self.tokenizer = tokenizer
        self.tokenizer_name = tokenizer_name
        self._device: str = get_torch_device_name()

    def _get_model(self, checkpoint: str) -> Vision2SeqModelProcessor:
        global _models_lock
        global _models

        # Ensure that only one thread is loading the model at a time
        with _models_lock:
            loaded_model_processor = _models[checkpoint]
            if loaded_model_processor is None:
                hlog(f"Loading model {checkpoint} and caching in memory...")
                torch_dtype: torch.dtype = torch.float16 if is_cuda_available() else torch.float32
                model = AutoModelForVision2Seq.from_pretrained(checkpoint, torch_dtype=torch_dtype).to(self._device)
                processor = AutoProcessor.from_pretrained(checkpoint)

                _models[checkpoint] = Vision2SeqModelProcessor(model, processor)
                loaded_model_processor = _models[checkpoint]

        assert loaded_model_processor is not None
        return loaded_model_processor

    def make_request(self, request: Request) -> RequestResult:
        assert request.model_deployment in _models, f"Not a valid model for this client: {request.model_deployment}"
        assert request.multimodal_prompt is not None, "Multimodal prompt is required"

        loaded_model_processor: Vision2SeqModelProcessor = self._get_model(request.model_deployment)
        model = loaded_model_processor.model
        processor = loaded_model_processor.processor

        generation_args: Dict[str, Any] = {
            "max_new_tokens": request.max_tokens,
        }

        image_paths: List[str] = []
        multimodal_prompt: List[Dict[str, str]] = []
        for media_object in request.multimodal_prompt.media_objects:
            if media_object.is_type("image") and media_object.location:
                image_paths.append(media_object.location)
                multimodal_prompt.append({"type": "image"})
            elif media_object.is_type(TEXT_TYPE):
                if media_object.text is None:
                    raise ValueError("MediaObject of text type has missing text field value")

                multimodal_prompt.append({"type": "text", "text": media_object.text})
            else:
                raise ValueError(f"Unrecognized MediaObject type {media_object.type}")

        completions: List[GeneratedOutput] = []
        with htrack_block(f"Generating for prompt: {request.multimodal_prompt.text}"):
            try:

                def do_it() -> Dict[str, Any]:
                    messages = [{"role": "user", "content": multimodal_prompt}]
                    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
                    inputs = processor(
                        text=[prompt] * request.num_completions,
                        images=[
                            [load_image(image_path) for image_path in image_paths]
                            for _ in range(request.num_completions)
                        ],
                        return_tensors="pt",
                    )
                    inputs = {k: v.to(self._device) for k, v in inputs.items()}

                    # Generate
                    generated_ids = model.generate(**inputs, **generation_args)
                    generated_texts: List[str] = processor.batch_decode(generated_ids, skip_special_tokens=True)
                    return {"output": generated_texts}

                # Include the prompt and model name in the cache key
                cache_key = CachingClient.make_cache_key(
                    raw_request={
                        "n": request.num_completions,
                        "model": request.model,
                        "prompt": generate_uid_for_multimodal_prompt(request.multimodal_prompt),
                        **generation_args,
                    },
                    request=request,
                )
                result, cached = self.cache.get(cache_key, wrap_request_time(do_it))
            except RuntimeError as model_error:
                return RequestResult(success=False, cached=False, error=str(model_error), completions=[], embedding=[])

            for text in result["output"]:
                hlog(f"Generated text: {text}")
                assert self.ASSISTANT_PREFIX in text, f"Expected {self.ASSISTANT_PREFIX} in the output"
                text = text.rpartition(self.ASSISTANT_PREFIX)[-1]
                hlog(f"Truncated: {text}")
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
