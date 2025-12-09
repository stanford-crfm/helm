import os.path
from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from helm.clients.client import CachingClient, generate_uid_for_multimodal_prompt
from helm.common.cache import CacheConfig
from helm.common.gpu_utils import get_torch_device_name
from helm.common.hierarchical_logger import hlog, htrack_block
from helm.common.media_object import TEXT_TYPE
from helm.common.request import (
    GeneratedOutput,
    Request,
    RequestResult,
    Token,
    wrap_request_time,
)


@dataclass(frozen=True)
class LoadedModelProcessor:
    model: Any
    processor: AutoProcessor


LOCAL_ROBOREWARD_QWEN3_4B = (
    "/nlp/scr4/nlp/crfm/text2image/text2image-rlhf/robotics/"
    "Qwen2.5-VL/qwen-vl-finetune/output_qwen3vl_4b_roboreward_validated/hf_checkpoints/step-3000"
)

LOCAL_ROBOREWARD_QWEN3_8B = (
    "/nlp/scr4/nlp/crfm/text2image/text2image-rlhf/robotics/"
    "Qwen2.5-VL/qwen-vl-finetune/output_qwen3vl_8b_roboreward_validated_127/hf_checkpoints/step-4200"
)


_models_lock: Lock = Lock()
_models: Dict[str, Optional[LoadedModelProcessor]] = {
    "Qwen/Qwen3-VL-4B-Instruct": None,
    "Qwen/Qwen3-VL-8B-Instruct": None,
    "Qwen/Qwen3-VL-30B-A3B-Instruct": None,
    "Qwen/Qwen3-VL-235B-A22B-Instruct": None,
    LOCAL_ROBOREWARD_QWEN3_4B: None,
    LOCAL_ROBOREWARD_QWEN3_8B: None,
}


class Qwen3VLMClient(CachingClient):
    def __init__(self, cache_config: CacheConfig):
        super().__init__(cache_config=cache_config)
        self._device: str = get_torch_device_name()

    def _get_model_name(self, helm_model_name: str) -> str:
        if helm_model_name == "qwen3-vl-4b-instruct":
            return "Qwen/Qwen3-VL-4B-Instruct"
        if helm_model_name == "qwen3-vl-8b-instruct":
            return "Qwen/Qwen3-VL-8B-Instruct"
        if helm_model_name == "qwen3-vl-30b-a3b-instruct":
            return "Qwen/Qwen3-VL-30B-A3B-Instruct"
        if helm_model_name == "qwen3-vl-235b-a22b-instruct":
            return "Qwen/Qwen3-VL-235B-A22B-Instruct"
        if helm_model_name == "qwen3-vl-4b-instruct-robo-reward":
            return LOCAL_ROBOREWARD_QWEN3_4B
        if helm_model_name == "qwen3-vl-8b-instruct-robo-reward":
            return LOCAL_ROBOREWARD_QWEN3_8B
        raise ValueError(f"Unhandled model name: {helm_model_name}")

    def _get_model(self, helm_model_name: str) -> LoadedModelProcessor:
        global _models_lock, _models

        model_name = self._get_model_name(helm_model_name)
        with _models_lock:
            loaded = _models[model_name]
            if loaded is None:
                hlog(f"Loading model {model_name} and caching in memory...")
                model = Qwen3VLForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    attn_implementation="flash_attention_2",
                ).eval()
                processor = AutoProcessor.from_pretrained(model_name)
                if not getattr(processor, "chat_template", None):
                    fallback = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
                    processor.chat_template = fallback.chat_template
                loaded = LoadedModelProcessor(model=model, processor=processor)
                _models[model_name] = loaded
        return loaded

    def make_request(self, request: Request) -> RequestResult:
        assert request.multimodal_prompt is not None, "Multimodal prompt is required"

        message_content: List[Dict[str, Any]] = []
        for media_object in request.multimodal_prompt.media_objects:
            if media_object.is_type("image") and media_object.location:
                message_content.append({"type": "image", "image": media_object.location})
            elif media_object.is_type("video") and media_object.location:
                message_content.append({"type": "video", "video": media_object.location})
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

                        inputs = processor.apply_chat_template(  # type: ignore
                            messages,
                            tokenize=True,
                            add_generation_prompt=True,
                            return_dict=True,
                            return_tensors="pt",
                        ).to(self._device)

                        generated_ids = model.generate(**inputs, **generation_args)
                        generated_ids_trimmed = [
                            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                        ]
                        output_text = processor.batch_decode(  # type: ignore
                            generated_ids_trimmed,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        )
                        completion = output_text[0]
                        tokens = completion.split()
                        hlog(f"Generated: {text_out}")
                        return {"output": (completion, tokens)}

                    raw_request = {
                        "completion_index": completion_index,
                        "model": request.model,
                        "prompt": generate_uid_for_multimodal_prompt(request.multimodal_prompt),
                        **generation_args,
                    }
                    local_model_name_or_path = self._get_model_name(request.model_engine)
                    if os.path.exists(local_model_name_or_path):
                        raw_request["local_model_name_or_path"] = local_model_name_or_path

                    cache_key = CachingClient.make_cache_key(
                        raw_request=raw_request,
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

