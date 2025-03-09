from typing import Dict, Optional

from helm.clients.azure_openai_client import AzureOpenAIClient
from helm.common.cache import CacheConfig
from helm.proxy.retry import NonRetriableException
from helm.tokenizers.tokenizer import Tokenizer


class StanfordHealthCareSHCOpenAIClient(AzureOpenAIClient):
    """
    Client for accessing OpenAI models hosted on Stanford Health Care's model API.

    Configure by setting the following in prod_env/credentials.conf:

    ```
    stanfordhealthcareEndpoint: https://your-domain-name/
    stanfordhealthcareApiKey: your-private-key
    ```
    """

    API_VERSION = "2024-08-01-preview"

    def __init__(
        self,
        tokenizer: Tokenizer,
        tokenizer_name: str,
        cache_config: CacheConfig,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
    ):
        if not api_key:
            raise NonRetriableException("Must provide API key through credentials.conf")
        super().__init__(
            tokenizer=tokenizer,
            tokenizer_name=tokenizer_name,
            cache_config=cache_config,
            api_key=api_key,
            endpoint=endpoint,
            api_version=StanfordHealthCareSHCOpenAIClient.API_VERSION,
            default_headers=default_headers,
        )
