import requests
from typing import Any, Dict, List

from helm.common.cache import CacheConfig
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.common.request import wrap_request_time, Request, RequestResult, Sequence, Token
from helm.common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
)
from helm.proxy.tokenizers.tokenizer import Tokenizer
from .client import CachingClient, truncate_sequence

try:
    from mistralai.client import MistralClient
    from mistralai.models.chat_completion import ChatMessage, ChatCompletionResponse
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, ["mistral"])


class MistralAIClient(CachingClient):
    """
    Client for Mistral API.
    """

    # Aliases to match HELM names to Model names in Mistral API
    _model_aliases: Dict[str, str] = {
        "mistral-7b-v0.1": "mistral-tiny",
    }

    def __init__(self, tokenizer: Tokenizer, tokenizer_name: str, cache_config: CacheConfig, api_key: str):
        super().__init__(cache_config=cache_config)
        self.api_key: str = api_key
        self.tokenizer = tokenizer
        self.tokenizer_name = tokenizer_name
        self._client = MistralClient(api_key=self.api_key)

    def _send_request(self, model_name: str, raw_request: Dict[str, Any]) -> Dict[str, Any]:
        messages = [ChatMessage(role="user", content=raw_request["prompt"])]

        chat_response: ChatCompletionResponse = self._client.chat(
            model=model_name,
            messages=messages,
            temperature=raw_request["temperature"],
            max_tokens=raw_request["max_tokens"],
            top_p=raw_request["top_p"],
            random_seed=raw_request["random_seed"],
            safe_prompt=False,  # Disable safe_prompt
        )

        return chat_response.dict()

    def make_request(self, request: Request) -> RequestResult:
        """Make a request"""
        raw_request = {
            "prompt": request.prompt,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "random_seed": request.random,
        }

        completions: List[Sequence] = []
        model_name: str = self._model_aliases.get(request.model_engine, request.model_engine)

        # `num_completions` is not supported, so instead make `num_completions` separate requests.
        for completion_index in range(request.num_completions):
            try:

                def do_it():
                    result: Dict[str, Any] = self._send_request(model_name, raw_request)
                    return result

                # We need to include the engine's name to differentiate among requests made for different model
                # engines since the engine name is not included in the request itself.
                # In addition, we want to make `request.num_completions` fresh
                # requests, cache key should contain the completion_index.
                # Echoing the original prompt is not officially supported by Mistral. We instead prepend the
                # completion with the prompt when `echo_prompt` is true, so keep track of it in the cache key.
                cache_key = CachingClient.make_cache_key(
                    {
                        "engine": model_name,
                        "completion_index": completion_index,
                        **raw_request,
                    },
                    request,
                )

                response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
            except (requests.exceptions.RequestException, AssertionError) as e:
                error: str = f"MistralClient error: {e}"
                return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])

            response_message: Dict[str, Any] = response["choices"][0]["message"]
            assert response_message["role"] == "assistant"
            response_text: str = response_message["content"]

            # The Mistral API doesn't support echo. If `echo_prompt` is true, combine the prompt and completion.
            text: str = request.prompt + response_text if request.echo_prompt else response_text
            # The Mistral API doesn't return us tokens or logprobs, so we tokenize ourselves.
            tokenization_result: TokenizationRequestResult = self.tokenizer.tokenize(
                TokenizationRequest(text, tokenizer=self.tokenizer_name)
            )

            # Log probs are not currently not supported by Mistral, so set to 0 for now.
            tokens: List[Token] = [Token(text=str(text), logprob=0) for text in tokenization_result.raw_tokens]

            completion = Sequence(text=response_text, logprob=0, tokens=tokens)
            sequence = truncate_sequence(completion, request, print_warning=True)
            completions.append(sequence)

        return RequestResult(
            success=True,
            cached=cached,
            request_time=response["request_time"],
            request_datetime=response["request_datetime"],
            completions=completions,
            embedding=[],
        )
