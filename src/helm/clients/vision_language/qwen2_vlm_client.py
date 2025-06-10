from threading import Lock
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

from helm.common.cache import CacheConfig
from helm.common.gpu_utils import get_torch_device_name
from helm.common.hierarchical_logger import hlog, htrack_block
from helm.common.media_object import TEXT_TYPE
from helm.common.request import Request, RequestResult, GeneratedOutput, Token
from helm.common.request import wrap_request_time
from helm.clients.client import CachingClient, generate_uid_for_multimodal_prompt


@dataclass(frozen=True)
class LoadedModelProcessor:
    model: Any
    processor: AutoProcessor


# Global cache for all models
_models_lock: Lock = Lock()
_models: Dict[str, Optional[LoadedModelProcessor]] = {
    "Qwen/Qwen2-VL-7B-Instruct": None,
    "Qwen/Qwen2-VL-72B-Instruct": None,
    "Qwen/Qwen2.5-VL-3B-Instruct": None,
    "Qwen/Qwen2.5-VL-7B-Instruct": None,
    "Qwen/Qwen2.5-VL-32B-Instruct": None,
    "Qwen/Qwen2.5-VL-72B-Instruct": None,
}


class Qwen2VLMClient(CachingClient):
    def __init__(self, cache_config: CacheConfig):
        super().__init__(cache_config=cache_config)
        self._device: str = get_torch_device_name()

    def _get_model_name(self, helm_model_name: str) -> str:
        if helm_model_name == "qwen2-vl-7b-instruct":
            return "Qwen/Qwen2-VL-7B-Instruct"
        elif helm_model_name == "qwen2-vl-72b-instruct":
            return "Qwen/Qwen2-VL-72B-Instruct"
        elif helm_model_name == "qwen2.5-vl-3b-instruct":
            return "Qwen/Qwen2.5-VL-3B-Instruct"
        elif helm_model_name == "qwen2.5-vl-7b-instruct":
            return "Qwen/Qwen2.5-VL-7B-Instruct"
        elif helm_model_name == "qwen2.5-vl-32b-instruct":
            return "Qwen/Qwen2.5-VL-32B-Instruct"
        elif helm_model_name == "qwen2.5-vl-72b-instruct":
            return "Qwen/Qwen2.5-VL-72B-Instruct"
        else:
            raise ValueError(f"Unhandled model name: {helm_model_name}")

    def _get_model(self, helm_model_name: str) -> LoadedModelProcessor:
        from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration

        global _models_lock, _models

        model_name = self._get_model_name(helm_model_name)
        with _models_lock:
            loaded = _models[model_name]
            if loaded is None:
                hlog(f"Loading model {model_name} and caching in memory...")
                # Use different loading routines depending on whether it's Qwen2.5 or Qwen2.
                if "2.5" in model_name:
                    # Qwen2.5: by default use torch_dtype="auto". You can enable flash_attention_2 if desired.
                    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        model_name,
                        torch_dtype=torch.bfloat16,
                        device_map="auto",
                        attn_implementation="flash_attention_2",
                    ).eval()
                else:
                    model = Qwen2VLForConditionalGeneration.from_pretrained(
                        model_name,
                        torch_dtype=torch.bfloat16,
                        device_map="auto",
                        attn_implementation="flash_attention_2",
                    ).eval()
                processor = AutoProcessor.from_pretrained(model_name)
                loaded = LoadedModelProcessor(model=model, processor=processor)
                _models[model_name] = loaded
        return loaded

    def make_request(self, request: Request) -> RequestResult:
        assert request.multimodal_prompt is not None, "Multimodal prompt is required"

        # Build messages by collating all media objects into a single "user" message.
        message_content = []
        for media_object in request.multimodal_prompt.media_objects:
            if media_object.is_type("image") and media_object.location:
                message_content.append({"type": "image", "image": media_object.location})
            elif media_object.is_type(TEXT_TYPE):
                if media_object.text is None:
                    raise ValueError("MediaObject of text type has missing text field value")
                message_content.append({"type": "text", "text": media_object.text})
            else:
                raise ValueError(f"Unrecognized MediaObject type {media_object.type}")

        messages = [{"role": "user", "content": message_content}]

        generation_args = {
            "max_new_tokens": request.max_tokens,
        }

        completions: List[GeneratedOutput] = []
        request_time: float = 0
        request_datetime: Optional[int] = None
        all_cached: bool = True

        with htrack_block(f"Generating for prompt: {request.multimodal_prompt.text}"):
            for completion_index in range(request.num_completions):
                try:

                    def do_it() -> Dict[str, Any]:
                        loaded = self._get_model(request.model_engine)
                        model = loaded.model
                        processor = loaded.processor

                        # Prepare text and vision inputs.
                        text = processor.apply_chat_template(  # type: ignore
                            messages, tokenize=False, add_generation_prompt=True
                        )
                        image_inputs, video_inputs = process_vision_info(messages)
                        inputs = processor(  # type: ignore
                            text=[text],
                            images=image_inputs,
                            videos=video_inputs,
                            padding=True,
                            return_tensors="pt",
                        ).to(self._device)

                        generated_ids = model.generate(**inputs, **generation_args)
                        # Remove the input prefix from outputs.
                        generated_ids_trimmed = [
                            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                        ]
                        output_text = processor.batch_decode(  # type: ignore
                            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                        )
                        # For simplicity, we split tokens by whitespace.
                        completion = output_text[0]
                        tokens = completion.split()
                        return {"output": (completion, tokens)}

                    cache_key = CachingClient.make_cache_key(
                        raw_request={
                            "completion_index": completion_index,
                            "model": request.model,
                            "prompt": generate_uid_for_multimodal_prompt(request.multimodal_prompt),
                            **generation_args,
                        },
                        request=request,
                    )
                    result, cached = self.cache.get(cache_key, wrap_request_time(do_it))
                except RuntimeError as model_error:
                    return RequestResult(
                        success=False,
                        cached=False,
                        error=str(model_error),
                        completions=[],
                        embedding=[],
                    )

                text_out, tokens = result["output"]
                completions.append(
                    GeneratedOutput(
                        text=text_out,
                        logprob=0,
                        tokens=[Token(text=str(token), logprob=0) for token in tokens],
                    )
                )
                hlog(f"Generated: {text_out}")
                request_time += result["request_time"]
                request_datetime = request_datetime or result.get("request_datetime")
                all_cached = all_cached and cached

        return RequestResult(
            success=True,
            cached=all_cached,
            request_time=request_time,
            request_datetime=request_datetime,
            completions=completions,
            embedding=[],
        )
