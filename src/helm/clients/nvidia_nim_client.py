from typing import Optional

from helm.clients.openai_client import OpenAIClient
from helm.common.cache import CacheConfig
from helm.common.request import Request


class NvidiaNimClient(OpenAIClient):

    BASE_URL = "https://integrate.api.nvidia.com/v1"

    def __init__(
        self,
        cache_config: CacheConfig,
        api_key: Optional[str] = None,
    ):
        super().__init__(
            cache_config=cache_config,
            api_key=api_key,
            org_id=None,
            base_url=NvidiaNimClient.BASE_URL,
        )

    def _get_model_for_request(self, request: Request) -> str:
        return request.model
