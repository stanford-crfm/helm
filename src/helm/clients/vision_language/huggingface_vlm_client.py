from threading import Lock
from typing import Any, Dict, List, Optional

from transformers import pipeline
from transformers.pipelines import ImageToTextPipeline

from helm.common.cache import CacheConfig
from helm.common.images_utils import open_image
from helm.common.media_object import TEXT_TYPE
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.common.request import Request, RequestResult, GeneratedOutput, Token
from helm.common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
)
from helm.common.request import wrap_request_time
from helm.clients.client import CachingClient, generate_uid_for_multimodal_prompt
from helm.tokenizers.tokenizer import Tokenizer

try:
    from PIL import Image
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, ["images"])


class HuggingFaceVLMClient(CachingClient):
    """
    General client for VLM models from HuggingFace.
    """

    _models_lock: Lock = Lock()
    _models: Dict[str, ImageToTextPipeline] = {}
    _models_aliases: Dict[str, str] = {
        "huggingface/llava-1.5-7b-hf": "llava-hf/llava-1.5-7b-hf",
        "huggingface/llava-1.5-13b-hf": "llava-hf/llava-1.5-13b-hf",
        "huggingface/bakLlava-v1-hf": "llava-hf/bakLlava-v1-hf",
        "huggingface/llava-v1.6-vicuna-7b-hf": "llava-hf/llava-v1.6-vicuna-7b-hf",
        "huggingface/llava-v1.6-vicuna-13b-hf": "llava-hf/llava-v1.6-vicuna-13b-hf",
        "huggingface/llava-v1.6-mistral-7b-hf": "llava-hf/llava-v1.6-mistral-7b-hf",
        "huggingface/llava-v1.6-34b-hf": "llava-hf/llava-v1.6-34b-hf",
        "huggingface/prometheus-vision-13b-v1.0-hf": "PahaII/prometheus-vision-13b-v1.0-hf",
    }

    def __init__(self, tokenizer: Tokenizer, tokenizer_name: str, cache_config: CacheConfig):
        super().__init__(cache_config=cache_config)
        self.tokenizer = tokenizer
        self.tokenizer_name = tokenizer_name

    def _get_model(self, model_name: str) -> ImageToTextPipeline:
        with self._models_lock:
            model_id: str = self._models_aliases.get(model_name, model_name)
            if model_id not in self._models:
                self._models[model_id] = pipeline("image-to-text", model=model_id, device_map="auto")
            return self._models[model_id]

    def make_request(self, request: Request) -> RequestResult:
        assert request.multimodal_prompt is not None, "Multimodal prompt is required"

        # Build the prompt
        prompt: str = ""
        image: Optional[Image.Image] = None
        for media_object in request.multimodal_prompt.media_objects:
            if media_object.is_type("image") and media_object.location:
                # TODO #2235: Figure out is fome HuggingFace models support multiple images
                if image is not None:
                    raise ValueError("Only one image is supported in the multimodal prompt")
                image = open_image(media_object.location)
            elif media_object.is_type(TEXT_TYPE):
                if media_object.text is None:
                    raise ValueError("MediaObject of text type has missing text field value")
                prompt += f"\n{media_object.text}"
            else:
                raise ValueError(f"Unsupported media object type: {media_object.type}")

        # Generate
        try:
            generation_args = {
                "max_new_tokens": request.max_tokens,
            }

            def do_it() -> Dict[str, Any]:
                model: ImageToTextPipeline = self._get_model(request.model_deployment)
                outputs = model(image, prompt=prompt, generate_kwargs=generation_args)
                return outputs[0]

            cache_key = CachingClient.make_cache_key(
                raw_request={
                    "model": request.model,
                    "prompt": generate_uid_for_multimodal_prompt(request.multimodal_prompt),
                    **generation_args,
                },
                request=request,
            )
            result, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except RuntimeError as e:
            return RequestResult(success=False, cached=False, error=str(e), completions=[], embedding=[])

        output: str = result["generated_text"]
        if "ASSISTANT: " in output:
            output = output.split("ASSISTANT: ")[1]
        tokenization_result: TokenizationRequestResult = self.tokenizer.tokenize(
            TokenizationRequest(output, tokenizer=self.tokenizer_name)
        )
        tokens: List[Token] = [Token(text=str(text), logprob=0) for text in tokenization_result.raw_tokens]
        completions: List[GeneratedOutput] = [GeneratedOutput(text=output, logprob=0, tokens=tokens)]
        return RequestResult(
            success=True,
            cached=cached,
            request_time=result["request_time"],
            completions=completions,
            embedding=[],
        )
