from typing import Optional

from helm.clients.openai_client import OpenAIClient
from helm.common.cache import CacheConfig


class YiChatClient(OpenAIClient):

    BASE_URL = "http://api.01ww.xyz/v1"

    def __init__(
        self,
        cache_config: CacheConfig,
        api_key: Optional[str] = None,
    ):
        super().__init__(
            cache_config=cache_config,
            api_key=api_key,
            org_id=None,
            base_url=YiChatClient.BASE_URL,
        )
