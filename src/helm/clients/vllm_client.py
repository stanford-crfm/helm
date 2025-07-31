from typing import Any, Dict, Optional

from helm.common.cache import CacheConfig
from helm.common.request import Request
from helm.clients.openai_client import OpenAIClient, OpenAILegacyCompletionsClient
from helm.tokenizers.tokenizer import Tokenizer


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
        return raw_request


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
