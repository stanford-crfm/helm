import os
from typing import Dict, Optional

from helm.clients.openai_client import OpenAIClient
from helm.common.cache import CacheConfig
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.proxy.retry import NonRetriableException
from helm.tokenizers.tokenizer import Tokenizer

try:
    from openai import AzureOpenAI
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, ["openai"])


class AzureOpenAIClient(OpenAIClient):
    API_VERSION = "2024-07-01-preview"

    def __init__(
        self,
        tokenizer: Tokenizer,
        tokenizer_name: str,
        cache_config: CacheConfig,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        api_version: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
    ):
        super().__init__(
            tokenizer=tokenizer, tokenizer_name=tokenizer_name, cache_config=cache_config, api_key="unused"
        )
        azure_endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        if not azure_endpoint:
            raise NonRetriableException("Must provide Azure endpoint through credentials.conf or AZURE_OPENAI_ENDPOINT")
        self.client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version or AzureOpenAIClient.API_VERSION,
            azure_endpoint=azure_endpoint,
            default_headers=default_headers,
        )
