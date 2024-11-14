from typing import Optional

from helm.clients.openai_client import OpenAIClient
from helm.common.cache import CacheConfig
from helm.tokenizers.tokenizer import Tokenizer


class YiChatClient(OpenAIClient):

    BASE_URL = "http://api.01ww.xyz/v1"

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
            base_url=YiChatClient.BASE_URL,
        )
