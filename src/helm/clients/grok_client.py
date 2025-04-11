from typing import Any, Dict, Optional

from helm.clients.openai_client import OpenAIClient
from helm.common.cache import CacheConfig
from helm.common.request import Request
from helm.tokenizers.tokenizer import Tokenizer


class GrokChatClient(OpenAIClient):

    BASE_URL = "https://api.x.ai/v1"

    _UNSUPPORTED_ARGUMENTS = ["presence_penalty", "frequency_penalty"]

    def __init__(
        self,
        tokenizer: Tokenizer,
        tokenizer_name: str,
        cache_config: CacheConfig,
        api_key: Optional[str] = None,
    ):
        super().__init__(
            tokenizer=tokenizer,
            tokenizer_name=tokenizer_name,
            cache_config=cache_config,
            api_key=api_key,
            org_id=None,
            base_url="https://api.x.ai/v1",
        )

    def _make_chat_raw_request(self, request: Request) -> Dict[str, Any]:
        raw_request = super()._make_chat_raw_request(request)
        for unsupported_argument in self._UNSUPPORTED_ARGUMENTS:
            if unsupported_argument in raw_request:
                del raw_request[unsupported_argument]
        return raw_request
