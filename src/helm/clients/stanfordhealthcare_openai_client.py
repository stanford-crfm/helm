from typing import Optional

from helm.clients.azure_openai_client import AzureOpenAIClient
from helm.common.cache import CacheConfig
from helm.proxy.retry import NonRetriableException
from helm.tokenizers.tokenizer import Tokenizer


class StanfordHealthCareOpenAIClient(AzureOpenAIClient):
    """
    Client for accessing OpenAI models hosted on Stanford Health Care's model API.

    Configure by setting the following in prod_env/credentials.conf:

    ```
    stanfordhealthcareEndpoint: https://your-domain-name/
    stanfordhealthcareApiKey: your-private-key
    ```
    """

    API_VERSION = "2023-05-15"
    CREDENTIAL_HEADER_NAME = "Ocp-Apim-Subscription-Key"

    def __init__(
        self,
        tokenizer: Tokenizer,
        tokenizer_name: str,
        cache_config: CacheConfig,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
    ):
        if not api_key:
            raise NonRetriableException("Must provide API key through credentials.conf")
        super().__init__(
            tokenizer=tokenizer,
            tokenizer_name=tokenizer_name,
            cache_config=cache_config,
            api_key="unused",
            endpoint=endpoint,
            api_version=StanfordHealthCareOpenAIClient.API_VERSION,
            default_headers={StanfordHealthCareOpenAIClient.CREDENTIAL_HEADER_NAME: api_key},
        )
