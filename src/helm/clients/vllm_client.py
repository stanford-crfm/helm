from typing import Any, Dict, List, Optional
from dataclasses import replace

from helm.clients.openai_client import OpenAIClient
from helm.common.cache import CacheConfig
from helm.common.hierarchical_logger import htrack_block, hlog
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.common.request import wrap_request_time, Request, RequestResult, GeneratedOutput, Token
from .client import truncate_sequence
from helm.tokenizers.tokenizer import Tokenizer

try:
    import openai
    from openai import OpenAI
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, ["openai"])


class VLLMClient(OpenAIClient):
    """Sends request to a vLLM server using the OpenAI-compatible API.

    See: https://docs.vllm.ai/en/latest/getting_started/quickstart.html#openai-compatible-server"""

    def __init__(
        self,
        tokenizer: Tokenizer,
        tokenizer_name: str,
        cache_config: CacheConfig,
        base_url: Optional[str] = None,
    ):
        super().__init__(
            tokenizer=tokenizer,
            tokenizer_name=tokenizer_name,
            cache_config=cache_config,
            api_key="EMPTY",
            org_id=None,
            base_url=base_url,
        )
        self.tokenizer = tokenizer
        self.tokenizer_name = tokenizer_name

    def _is_chat_model_engine(self, model_engine: str) -> bool:
        # Only support vLLM completion models for now.
        return False

    def _get_model_for_request(self, request: Request) -> str:
        # The `model` parameter for vLLM should be the whole model name including the creator organization,
        # unlike OpenAI which only uses the model engine.
        return request.model

    def _to_raw_completion_request(self, request: Request) -> Dict[str, Any]:
        raw_request = super()._to_raw_completion_request(request)
        # This avoids the error: best_of must be 1 when using greedy sampling
        if "best_of" in raw_request and raw_request["best_of"] > 1:
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

        with htrack_block(f"Prompt: {request.prompt}"):
            for raw_completion in response["choices"]:
                sequence_logprob = 0
                tokens: List[Token] = []
                completion = GeneratedOutput(
                    text=raw_completion["text"],
                    logprob=sequence_logprob,
                    tokens=tokens,
                    finish_reason={"reason": raw_completion["finish_reason"]},
                )
                hlog(completion.text)

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
