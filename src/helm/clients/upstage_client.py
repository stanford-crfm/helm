from helm.clients.openai_client import OpenAIClient
from helm.common.cache import CacheConfig
from helm.tokenizers.tokenizer import Tokenizer


class UpstageChatClient(OpenAIClient):
    """Sends request to a Upstage model using a OpenAI-compatible Chat API."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        tokenizer_name: str,
        cache_config: CacheConfig,
        api_key: str,
    ):
        super().__init__(
            tokenizer=tokenizer,
            tokenizer_name=tokenizer_name,
            cache_config=cache_config,
            api_key=api_key,
            org_id=None,
            base_url="https://api.upstage.ai/v1/solar",
        )
