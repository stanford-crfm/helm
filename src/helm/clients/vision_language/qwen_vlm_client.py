from threading import Lock
from typing import Any, Dict, List, Optional

from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

from helm.common.cache import CacheConfig
from helm.common.gpu_utils import get_torch_device_name
from helm.common.hierarchical_logger import hlog, htrack_block
from helm.common.media_object import TEXT_TYPE
from helm.common.request import Request, RequestResult, GeneratedOutput, Token
from helm.common.request import wrap_request_time
from helm.clients.client import CachingClient, generate_uid_for_multimodal_prompt


@dataclass(frozen=True)
class LoadedQwenModelProcessor:
    """Loaded model and processor for Qwen."""

    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer


_models_lock: Lock = Lock()
_models: Dict[str, Optional[LoadedQwenModelProcessor]] = {
    "Qwen/Qwen-VL": None,
    "Qwen/Qwen-VL-Chat": None,
}


class QwenVLMClient(CachingClient):
    """
    From https://huggingface.co/Qwen/Qwen-VL,
    Qwen-VL (Qwen Large Vision Language Model) is the visual multimodal version of the large model series,
    Qwen (abbr. Tongyi Qianwen), proposed by Alibaba Cloud. Qwen-VL accepts image, text, and bounding box
    as inputs, outputs text and bounding box.
    Alibaba released Qwen-VL and Qwen-VL-Chat, which is a chatbot model based on Qwen-VL.

    Paper: https://arxiv.org/abs/2308.12966
    """

    END_OF_TEXT_TOKEN: str = "<|endoftext|>"

    def __init__(self, cache_config: CacheConfig):
        super().__init__(cache_config=cache_config)
        self._device: str = get_torch_device_name()

    def _get_model(self, helm_model_name: str) -> LoadedQwenModelProcessor:
        global _models_lock
        global _models

        model_name: str
        if helm_model_name == "qwen-vl-chat":
            model_name = "Qwen/Qwen-VL-Chat"
        elif helm_model_name == "qwen-vl":
            model_name = "Qwen/Qwen-VL"
        else:
            raise ValueError(f"Unhandled model name: {helm_model_name}")

        # Ensure that only one thread is loading the model at a time
        with _models_lock:
            loaded_model_processor = _models[model_name]
            if loaded_model_processor is None:
                hlog(f"Loading model {model_name} and caching in memory...")
                model = AutoModelForCausalLM.from_pretrained(
                    model_name, device_map=self._device, trust_remote_code=True, bf16=True
                ).eval()
                if model_name == "Qwen/Qwen-VL-Chat":
                    model.generation_config = GenerationConfig.from_pretrained(model_name, trust_remote_code=True)
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                _models[model_name] = LoadedQwenModelProcessor(model, tokenizer)
                loaded_model_processor = _models[model_name]

        assert loaded_model_processor is not None
        return loaded_model_processor

    def make_request(self, request: Request) -> RequestResult:
        assert request.multimodal_prompt is not None, "Multimodal prompt is required"

        loaded_model_processor: LoadedQwenModelProcessor = self._get_model(request.model_engine)
        model = loaded_model_processor.model
        tokenizer = loaded_model_processor.tokenizer

        generation_args = {
            "max_length": request.max_tokens,
        }

        query: List[Dict[str, str]] = []
        prompt_text: str = ""

        image_index: int = 1
        for media_object in request.multimodal_prompt.media_objects:
            if media_object.is_type("image") and media_object.location:
                query.append({"image": media_object.location})
                prompt_text += f"Picture {image_index}: <img>{media_object.location}</img>\n"
                image_index += 1
            elif media_object.is_type(TEXT_TYPE):
                if media_object.text is None:
                    raise ValueError("MediaObject of text type has missing text field value")

                query.append({"text": media_object.text})
                prompt_text += media_object.text
            else:
                raise ValueError(f"Unrecognized MediaObject type {media_object.type}")

        completions: List[GeneratedOutput] = []
        request_time: float = 0
        request_datetime: Optional[int] = None
        all_cached: bool = True

        with htrack_block(f"Generating for prompt: {prompt_text}"):
            for completion_index in range(request.num_completions):
                try:

                    def do_it() -> Dict[str, Any]:
                        if request.model_engine == "qwen-vl-chat":
                            completion, _ = model.chat(tokenizer, query=tokenizer.from_list_format(query), history=None)
                        else:
                            inputs = tokenizer(tokenizer.from_list_format(query), return_tensors="pt")
                            inputs = inputs.to(self._device)
                            pred = model.generate(**inputs, **generation_args)
                            completion = tokenizer.decode(pred.cpu()[0], skip_special_tokens=False)

                        tokens: List[str] = tokenizer.tokenize(completion)
                        return {"output": (completion, tokens)}

                    # Include the prompt and model name in the cache key
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
                        success=False, cached=False, error=str(model_error), completions=[], embedding=[]
                    )

                text, tokens = result["output"]

                # Truncate the output text as the original Qwen includes the prompt in the output sequence
                if request.model_engine == "qwen-vl":
                    text = text[len(prompt_text) :]
                    text = text.replace(self.END_OF_TEXT_TOKEN, "")
                    hlog(f"Truncated: {text}")

                # Tokenize truncated text to get the list of tokens
                completions.append(
                    GeneratedOutput(
                        text=text, logprob=0, tokens=[Token(text=str(token), logprob=0) for token in tokens]
                    )
                )

                request_time += result["request_time"]
                # Use the datetime from the first completion because that's when the request was fired
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
