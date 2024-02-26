from typing import Optional

from helm.common.cache import CacheConfig
from helm.common.request import Request
from helm.proxy.clients.openai_client import OpenAIClient
from helm.proxy.tokenizers.tokenizer import Tokenizer


class VLLMClient(OpenAIClient):
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
        # Only support completion models for now.
        return False

    def _get_model_for_request(self, request: Request) -> str:
        # The `model` parameter for vLLM should be the whole model name including the creator organization,
        # unlike OpenAI which only uses the model engine.
        return request.model
