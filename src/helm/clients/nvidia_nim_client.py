from typing import Optional

from helm.clients.openai_client import OpenAIClient
from helm.common.cache import CacheConfig
from helm.common.request import Request
from helm.tokenizers.tokenizer import Tokenizer


class NvidiaNimClient(OpenAIClient):

    BASE_URL = "https://integrate.api.nvidia.com/v1"

    def __init__(
        self,
        tokenizer: Tokenizer,
        tokenizer_name: str,
        cache_config: CacheConfig,
        api_key: Optional[str] = None,
    ):
        self.tokenizer = tokenizer
        self.tokenizer_name = tokenizer_name
        super().__init__(
            tokenizer=tokenizer,
            tokenizer_name=tokenizer_name,
            cache_config=cache_config,
            api_key=api_key,
            org_id=None,
            base_url=NvidiaNimClient.BASE_URL,
        )

    def _get_model_for_request(self, request: Request) -> str:
        return request.model
