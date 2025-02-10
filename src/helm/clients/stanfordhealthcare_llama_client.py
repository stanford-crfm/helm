from typing import Optional

from helm.clients.openai_client import OpenAIClient
from helm.common.cache import CacheConfig
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.proxy.retry import NonRetriableException
from helm.tokenizers.tokenizer import Tokenizer

try:
    from openai import OpenAI
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, ["openai"])


class StanfordHealthCareLlamaClient(OpenAIClient):
    """
    Client for accessing Llama models hosted on Stanford Health Care's model API.

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
        model_name: str,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
    ):
        super().__init__(
            tokenizer=tokenizer, tokenizer_name=tokenizer_name, cache_config=cache_config, api_key="unused"
        )
        if not endpoint:
            raise NonRetriableException("Must provide endpoint through credentials.conf")
        if not api_key:
            raise NonRetriableException("Must provide API key through credentials.conf")
        # Guess the base URL part based on the model name
        # Maybe make this configurable instead?
        base_url_part = model_name.split("/")[1].lower().removesuffix("-instruct").replace("-", "").replace(".", "")

        base_url = f"{endpoint.strip('/')}/{base_url_part}/v1"
        self.client = OpenAI(
            api_key="dummy",
            base_url=base_url,
            default_headers={StanfordHealthCareLlamaClient.CREDENTIAL_HEADER_NAME: api_key},
        )
