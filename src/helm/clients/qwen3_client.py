from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from helm.common.cache import CacheConfig
from helm.common.gpu_utils import get_torch_device_name
from helm.common.hierarchical_logger import hexception, hlog, htrack_block
from helm.common.request import Request, RequestResult, GeneratedOutput, Token, wrap_request_time
from helm.clients.client import CachingClient


QWEN3_MODEL_MAP: Dict[str, str] = {
    "qwen3-4b-instruct-2507": "Qwen/Qwen3-4B-Instruct-2507",
    "openr1-distill-qwen3-1.7b-math": "teetone/OpenR1-Distill-Qwen3-1.7B-Math",
}


@dataclass(frozen=True)
class LoadedQwen3Model:
    model: Any
    tokenizer: Any


_models_lock: Lock = Lock()
_loaded_models: Dict[str, Optional[LoadedQwen3Model]] = {hf_name: None for hf_name in QWEN3_MODEL_MAP.values()}


class Qwen3Client(CachingClient):
    """Client for Qwen3 text-only Hugging Face models."""

    def __init__(self, cache_config: CacheConfig):
        super().__init__(cache_config=cache_config)
        self._device: str = get_torch_device_name()

    def _get_model_name(self, helm_model_name: str) -> str:
        if helm_model_name in QWEN3_MODEL_MAP:
            return QWEN3_MODEL_MAP[helm_model_name]
        raise ValueError(f"Unhandled Qwen3 model name: {helm_model_name}")

    def _get_model(self, helm_model_name: str) -> LoadedQwen3Model:
        hf_name = self._get_model_name(helm_model_name)
        force_download = "openr1" in hf_name.lower()
        with _models_lock:
            loaded = _loaded_models[hf_name]
            if loaded is None:
                hlog(f"Loading Qwen3 model {hf_name}...")
                tokenizer = AutoTokenizer.from_pretrained(
                    hf_name, trust_remote_code=True, force_download=force_download
                )
                model = AutoModelForCausalLM.from_pretrained(
                    hf_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else "auto",
                    device_map="auto",
                    force_download=force_download,
                )
                loaded = LoadedQwen3Model(model=model.eval(), tokenizer=tokenizer)
                _loaded_models[hf_name] = loaded
        return loaded

    def make_request(self, request: Request) -> RequestResult:
        # Build messages
        if request.messages is not None:
            messages = request.messages
        else:
            messages = [{"role": "user", "content": request.prompt}]

        raw_request: Dict[str, Any] = {
            "engine": request.model_engine,
            "messages": messages,
            "max_new_tokens": request.max_tokens,
        }

        # Load model/tokenizer once per request
        loaded = self._get_model(request.model_engine)
        tokenizer = loaded.tokenizer
        model = loaded.model

        def do_it() -> Dict[str, Any]:
            chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([chat_text], return_tensors="pt").to(model.device)

            with htrack_block(f"Generating for prompt={chat_text}"):
                output = model.generate(
                    **inputs,
                    max_new_tokens=request.max_tokens,
                    do_sample=False,
                    use_cache=True,
                    return_dict_in_generate=True,
                )
                sequences = output.sequences
                prompt_length = inputs.input_ids.shape[1]
                generated = sequences[:, prompt_length:]

                decoded = tokenizer.batch_decode(
                    generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                # Print last 100 characters of the decoded text
                hlog(f"Decoded: {decoded[0][-100:]}")
                return {"completions": decoded}

        try:
            cache_key = CachingClient.make_cache_key(raw_request, request)
            result, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except RuntimeError as model_error:
            hexception(model_error)
            return RequestResult(
                success=False,
                cached=False,
                error=str(model_error),
                completions=[],
                embedding=[],
            )

        completions: List[GeneratedOutput] = []
        for text in result["completions"]:
            token_strs: List[str] = tokenizer.tokenize(text)
            completions.append(
                GeneratedOutput(
                    text=text,
                    logprob=0.0,
                    tokens=[Token(text=token, logprob=0.0) for token in token_strs],
                )
            )

        return RequestResult(
            success=True,
            cached=cached,
            request_time=result["request_time"],
            request_datetime=result.get("request_datetime"),
            completions=completions,
            embedding=[],
        )

