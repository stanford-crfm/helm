from typing import Any, Dict, Optional

from helm.common.cache import CacheConfig
from helm.common.request import Request
from helm.clients.openai_client import OpenAILegacyCompletionsClient
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

    def _get_model_for_request(self, request: Request) -> str:
        # The `model` parameter for vLLM should be the whole model name including the creator organization,
        # unlike OpenAI which only uses the model engine.
        return request.model

    def _to_raw_completion_request(self, request: Request) -> Dict[str, Any]:
        raw_request = super()._to_raw_completion_request(request)
        # This avoids the error: best_of must be 1 when using greedy sampling
        if "best_of" in raw_request and raw_request["best_of"] > 1:
            raw_request["best_of"] = 1
        return raw_request
