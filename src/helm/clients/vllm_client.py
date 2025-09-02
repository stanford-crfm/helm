from threading import Lock
from typing import Any, Dict, List, Optional
from dataclasses import replace

from helm.clients.openai_client import OpenAIClient, OpenAILegacyCompletionsClient
from helm.common.cache import CacheConfig
from helm.common.hierarchical_logger import htrack_block
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.common.request import wrap_request_time, Request, RequestResult, GeneratedOutput, Token
from .client import OpenAIClient, truncate_sequence, CachingClient
from helm.tokenizers.tokenizer import Tokenizer

try:
    import openai
    from openai import OpenAI
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, ["openai"])


class VLLMClient(OpenAILegacyCompletionsClient):
    """Sends request to a vLLM server using the OpenAI-compatible API.

    Only supports the legacy Text Completions API, rather than the Chat Completions API.

    See: https://docs.vllm.ai/en/latest/getting_started/quickstart.html#openai-compatible-server"""

    def __init__(
        self,
        tokenizer: Tokenizer,
        tokenizer_name: str,
        cache_config: CacheConfig,
        base_url: Optional[str] = None,
        vllm_model_name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            tokenizer=tokenizer,
            tokenizer_name=tokenizer_name,
            cache_config=cache_config,
            api_key="EMPTY",
            org_id=None,
            base_url=base_url,
            openai_model_name=vllm_model_name,
            **kwargs,
        )
        self.tokenizer = tokenizer
        self.tokenizer_name = tokenizer_name
        self.vllm_model_name = vllm_model_name

    def _to_raw_completion_request(self, request: Request) -> Dict[str, Any]:
        raw_request = super()._to_raw_completion_request(request)
        # This avoids the error: best_of must be 1 when using greedy sampling
        if (
            "temperature" in raw_request
            and raw_request["temperature"] == 0.0
            and "best_of" in raw_request
            and raw_request["best_of"] > 1
        ):
            raw_request["best_of"] = 1

        # logprobs is not supported by TPUs
        raw_request.pop("logprobs", None)

        return raw_request

    def _make_completion_request(self, request: Request) -> RequestResult:
        raw_request = self._to_raw_completion_request(request)

        def do_it() -> Dict[str, Any]:
            return self.client.completions.create(**raw_request).model_dump(mode="json")

        try:
            cache_key = self._get_cache_key(raw_request, request)
            response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except openai.OpenAIError as e:
            error: str = f"OpenAI error: {e}"
            return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])

        completions: List[GeneratedOutput] = []
        for raw_completion in response["choices"]:
            sequence_logprob = 0
            tokens: List[Token] = []
            completion = GeneratedOutput(
                text=raw_completion["text"],
                logprob=sequence_logprob,
                tokens=tokens,
                finish_reason={"reason": raw_completion["finish_reason"]},
            )

            # OpenAI sends us back tokens past the end of text token,
            # so we need to manually truncate the list of tokens.
            # TODO: filed an issue with their support to check what the expected behavior here is.
            completion = truncate_sequence(
                completion, replace(request, stop_sequences=request.stop_sequences + [OpenAIClient.END_OF_TEXT])
            )
            completions.append(completion)

        return RequestResult(
            success=True,
            cached=cached,
            request_time=response["request_time"],
            request_datetime=response.get("request_datetime"),
            completions=completions,
            embedding=[],
        )


_models_lock: Lock = Lock()
_models: Dict[str, Any] = {}


class LocalVLLMClient(CachingClient):
    @staticmethod
    def get_model(request: Request) -> Dict:
        from vllm import LLM

        if request.model not in _models:
            with _models_lock:
                if request.model not in _models:
                    with htrack_block(f"Loading model {request.model}"):
                        _models[request.model] = LLM(model=request.model, enforce_eager=False, trust_remote_code=True)
        return _models[request.model]

    def __init__(self, cache_config: CacheConfig):
        super().__init__(cache_config=cache_config)

    def make_request(self, request: Request) -> RequestResult:
        raw_request = {
            "prompt": request.prompt,
            "echo": request.echo_prompt,
            "max_tokens": request.max_tokens,
            "model": request.model_engine,
            "n": request.num_completions,
            "stop": request.stop_sequences or None,
            "temperature": request.temperature,
            "top_p": request.top_p,
        }
        cache_key = CachingClient.make_cache_key(raw_request, request)
        model = self.get_model(request)

        try:

            def do_it():
                from vllm import SamplingParams

                sampling_params = SamplingParams(
                    n=request.num_completions,
                    presence_penalty=request.presence_penalty,
                    frequency_penalty=request.frequency_penalty,
                    stop=request.stop_sequences,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    # Currently, top-p sampling is disabled. `top_p` should be 1.0.
                    top_p=1.0,
                )
                outputs = model.generate(request.prompt, sampling_params)
                completions: List[str] = []
                for output in outputs:
                    generated_text: str = output.outputs[0].text
                    completions.append(generated_text)
                return {"completions": completions}

            response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except RuntimeError as e:
            error: str = f"vLLM inference error: {e}"
            return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])

        completions: List[GeneratedOutput] = []
        for completion in response["completions"]:
            completion = GeneratedOutput(
                text=completion,
                logprob=0,
                tokens=[],
                finish_reason={"reason": "unknown"},
            )
            completion = truncate_sequence(completion, request)
            completions.append(completion)

        return RequestResult(
            success=True,
            cached=cached,
            request_time=response["request_time"],
            completions=completions,
            embedding=[],
        )


class VLLMChatClient(OpenAIClient):
    """Sends request to a vLLM server using the OpenAI-compatible API.

    Only uses the Chat Completions API.

    See: https://docs.vllm.ai/en/latest/getting_started/quickstart.html#openai-compatible-server"""

    def __init__(
        self,
        tokenizer: Tokenizer,
        tokenizer_name: str,
        cache_config: CacheConfig,
        base_url: Optional[str] = None,
        vllm_model_name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            tokenizer=tokenizer,
            tokenizer_name=tokenizer_name,
            cache_config=cache_config,
            api_key="EMPTY",
            org_id=None,
            base_url=base_url,
            openai_model_name=vllm_model_name,
            **kwargs,
        )
        self.tokenizer = tokenizer
        self.tokenizer_name = tokenizer_name
        self.vllm_model_name = vllm_model_name
