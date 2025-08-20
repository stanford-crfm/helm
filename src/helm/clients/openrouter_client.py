import os
from typing import Optional
from helm.clients.openai_client import OpenAIClient
from helm.common.cache import CacheConfig
from helm.tokenizers.tokenizer import Tokenizer


class OpenRouterClient(OpenAIClient):
    def __init__(
        self,
        tokenizer_name: str,
        tokenizer: Tokenizer,
        cache_config: CacheConfig,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        output_processor: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1/"
        super().__init__(
            tokenizer,
            tokenizer_name,
            cache_config=cache_config,
            output_processor=output_processor,
            base_url=self.base_url,
            api_key=self.api_key,
        )
        self.model_name = model_name

    def _get_model_for_request(self, request):
        return self.model_name or request.model
