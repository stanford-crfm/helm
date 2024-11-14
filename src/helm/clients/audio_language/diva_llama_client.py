import threading
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import numpy as np
from transformers import AutoModel, PreTrainedModel

from helm.clients.client import CachingClient
from helm.common.cache import CacheConfig
from helm.common.media_object import TEXT_TYPE
from helm.common.request import (
    GeneratedOutput,
    Request,
    RequestResult,
    wrap_request_time,
)
from helm.common.audio_utils import get_array_from_audio_file
from helm.proxy.retry import NonRetriableException


_LOCK = threading.Lock()
_PRE_TRAINED_MODEL: Optional[PreTrainedModel] = None


def _get_pre_trained_model(model_name: str, **kwargs) -> PreTrainedModel:
    global _LOCK
    global _PRE_TRAINED_MODEL
    with _LOCK:
        if _PRE_TRAINED_MODEL is None:
            _PRE_TRAINED_MODEL = AutoModel.from_pretrained(model_name, **kwargs)
    return _PRE_TRAINED_MODEL


class DivaLlamaRequest(TypedDict):
    """Cache key for DivaLlamaClient"""

    model: str
    media_objects: List[Dict[str, Any]]


class DivaLlamaClient(CachingClient):
    SAMPLE_RATE = 16000

    def __init__(
        self,
        cache_config: CacheConfig,
        **kwargs,
    ):
        super().__init__(cache_config)
        self.pre_trained_model = _get_pre_trained_model("WillHeld/DiVA-llama-3-v0-8b", trust_remote_code=True, **kwargs)

    @staticmethod
    def _get_generate_input(request: Request) -> Tuple[np.ndarray, Optional[str]]:
        if request.prompt:
            raise NonRetriableException("request.prompt must be empty for DivaLlamaClient")
        if request.embedding:
            raise NonRetriableException("request.embedding must be empty for DivaLlamaClient")
        if request.messages:
            raise NonRetriableException("request.messages must be empty for DivaLlamaClient")
        if request.multimodal_prompt is None:
            raise NonRetriableException("request.multimodal_prompt must not be None for DivaLlamaClient")
        text_input: Optional[str] = None
        audio_input: Optional[np.ndarray] = None
        for media_object in request.multimodal_prompt.media_objects:
            if media_object.is_type("audio"):
                if audio_input is not None:
                    raise NonRetriableException(
                        "Only one audio object allowed in request.multimodal_prompt.media_objects"
                    )
                assert media_object.location
                audio_input = get_array_from_audio_file(media_object.location, DivaLlamaClient.SAMPLE_RATE)
            elif media_object.is_type(TEXT_TYPE):
                if text_input is not None:
                    raise NonRetriableException(
                        "Only one text object allowed in request.multimodal_prompt.media_objects"
                    )
                assert media_object.text is not None
                text_input = media_object.text
            else:
                raise NonRetriableException(f"Unsupported media content type type: {media_object.content_type}")
        if audio_input is None:
            raise NonRetriableException(
                "Expected a single audio object allowed in request.multimodal_prompt.media_objects"
            )
        return audio_input, text_input

    def make_request(self, request: Request) -> RequestResult:
        assert request.multimodal_prompt is not None
        raw_request: DivaLlamaRequest = {
            "model": request.model,
            "media_objects": [media_object.to_dict() for media_object in request.multimodal_prompt.media_objects],
        }

        try:

            def do_it() -> Dict[str, Any]:
                with _LOCK:
                    audio_input, text_input = DivaLlamaClient._get_generate_input(request)
                    if text_input is None:
                        return {"completions": self.pre_trained_model.generate([audio_input])}
                    else:
                        return {"completions": self.pre_trained_model.generate([audio_input], [text_input])}

            cache_key = CachingClient.make_cache_key(raw_request, request)
            response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except Exception as e:  # Do something if error is encountered.
            error: str = f"HuggingFace error: {e}"
            return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])

        generated_output = GeneratedOutput(text=response["completions"][0], logprob=0, tokens=[])

        return RequestResult(
            success=True,
            cached=cached,
            request_time=response["request_time"],
            request_datetime=response.get("request_datetime"),
            completions=[generated_output],
            embedding=[],
        )
