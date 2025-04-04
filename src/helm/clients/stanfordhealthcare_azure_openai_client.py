from typing import Optional

from helm.clients.azure_openai_client import AzureOpenAIClient
from helm.common.cache import CacheConfig
from helm.proxy.retry import NonRetriableException
from helm.tokenizers.tokenizer import Tokenizer


class StanfordHealthCareAzureOpenAIClient(AzureOpenAIClient):
    """
    Client for accessing OpenAI models hosted on Stanford Health Care's model API.

    Configure by setting the following in prod_env/credentials.conf:

    ```
    stanfordhealthcareEndpoint: https://your-domain-name/
    stanfordhealthcareApiKey: your-private-key
    ```
    """

    CREDENTIAL_HEADER_NAME = "Ocp-Apim-Subscription-Key"

    def __init__(
        self,
        tokenizer: Tokenizer,
        tokenizer_name: str,
        cache_config: CacheConfig,
        openai_model_name: str,
        api_version: str,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        if not api_key:
            raise NonRetriableException("Must provide API key through credentials.conf")
        if base_url:
            base_url = base_url.format(endpoint=endpoint)
            super().__init__(
                tokenizer=tokenizer,
                tokenizer_name=tokenizer_name,
                cache_config=cache_config,
                api_key="unused",
                base_url=base_url,
                azure_openai_deployment_name=openai_model_name,
                api_version=api_version,
                default_headers={StanfordHealthCareAzureOpenAIClient.CREDENTIAL_HEADER_NAME: api_key},
            )
        else:
            super().__init__(
                tokenizer=tokenizer,
                tokenizer_name=tokenizer_name,
                cache_config=cache_config,
                api_key="unused",
                endpoint=endpoint,
                azure_openai_deployment_name=openai_model_name,
                api_version=api_version,
                default_headers={StanfordHealthCareAzureOpenAIClient.CREDENTIAL_HEADER_NAME: api_key},
            )
