import requests
from typing import Any, Dict, List, Optional, TypedDict

from helm.proxy.retry import NonRetriableException
from helm.common.cache import CacheConfig
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.common.request import wrap_request_time, Request, RequestResult, GeneratedOutput
from helm.tokenizers.tokenizer import Tokenizer
from .client import CachingClient, truncate_and_tokenize_response_text

try:
    from mistralai.client import MistralClient
    from mistralai.models.chat_completion import ChatMessage, ChatCompletionResponse
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, ["mistral"])


class MistralAIRequest(TypedDict):
    """Data passed between make_request and _send_request. Used as the cache key."""

    model: str
    prompt: str
    max_tokens: int
    temperature: float
    top_p: float
    random_seed: Optional[int]


class MistralAIClient(CachingClient):
    """
    Client for Mistral API.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        tokenizer_name: str,
        cache_config: CacheConfig,
        api_key: str,
        mistral_model: Optional[str] = None,
    ):
        super().__init__(cache_config=cache_config)
        self.api_key: str = api_key
        self.tokenizer = tokenizer
        self.tokenizer_name = tokenizer_name
        self._client = MistralClient(api_key=self.api_key)
        self.mistral_model = mistral_model

    def _send_request(self, raw_request: MistralAIRequest) -> Dict[str, Any]:
        messages = [ChatMessage(role="user", content=raw_request["prompt"])]

        chat_response: ChatCompletionResponse = self._client.chat(
            model=raw_request["model"],
            messages=messages,
            temperature=raw_request["temperature"],
            max_tokens=raw_request["max_tokens"],
            top_p=raw_request["top_p"],
            random_seed=raw_request["random_seed"],
            safe_prompt=False,  # Disable safe_prompt
        )
        # Documentation: "If mode is 'json', the output will only contain JSON serializable types."
        # Source: https://docs.pydantic.dev/latest/api/base_model/#pydantic.BaseModel.model_dump
        #
        # We need to ensure that the output only contains JSON serializable types because the output
        # will be serialized for storage in the cache.
        return chat_response.model_dump(mode="json")

    def _get_random_seed(self, request: Request, completion_index: int) -> Optional[int]:
        if request.random is None and completion_index == 0:
            return None

        # Treat the user's request.random as an integer for the random seed.
        try:
            request_random_seed = int(request.random) if request.random is not None else 0
        except ValueError:
            raise NonRetriableException("MistralAIClient only supports integer values for request.random")

        # A large prime is used so that the resulting values are unlikely to collide
        # with request.random values chosen by the user.
        fixed_large_prime = 1911011
        completion_index_random_seed = completion_index * fixed_large_prime

        return request_random_seed + completion_index_random_seed

    def make_request(self, request: Request) -> RequestResult:
        """Make a request"""
        completions: List[GeneratedOutput] = []

        # `num_completions` is not supported, so instead make `num_completions` separate requests.
        for completion_index in range(request.num_completions):
            try:
                raw_request: MistralAIRequest = {
                    "model": self.mistral_model or request.model_engine,
                    "prompt": request.prompt,
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "random_seed": self._get_random_seed(request, completion_index),
                }

                def do_it() -> Dict[str, Any]:
                    result: Dict[str, Any] = self._send_request(raw_request)
                    return result

                # We need to include the engine's name to differentiate among requests made for different model
                # engines since the engine name is not included in the request itself.
                # In addition, we want to make `request.num_completions` fresh
                # requests, cache key should contain the completion_index.
                # Echoing the original prompt is not officially supported by Mistral. We instead prepend the
                # completion with the prompt when `echo_prompt` is true, so keep track of it in the cache key.
                cache_key = CachingClient.make_cache_key(raw_request, request)

                response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
            except (requests.exceptions.RequestException, AssertionError) as e:
                error: str = f"MistralClient error: {e}"
                return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])

            response_message: Dict[str, Any] = response["choices"][0]["message"]
            assert response_message["role"] == "assistant"
            response_text: str = response_message["content"]

            # The Mistral API doesn't support echo. If `echo_prompt` is true, combine the prompt and completion.
            text: str = request.prompt + response_text if request.echo_prompt else response_text
            sequence = truncate_and_tokenize_response_text(text, request, self.tokenizer, self.tokenizer_name)
            completions.append(sequence)

        return RequestResult(
            success=True,
            cached=cached,
            request_time=response["request_time"],
            request_datetime=response["request_datetime"],
            completions=completions,
            embedding=[],
        )
