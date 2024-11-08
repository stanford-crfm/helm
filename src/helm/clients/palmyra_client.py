# mypy: check_untyped_defs = False
import json
import requests
from typing import Any, Dict, List

from helm.clients.openai_client import OpenAIClient
from helm.common.cache import CacheConfig
from helm.common.hierarchical_logger import hlog
from helm.common.request import wrap_request_time, Request, RequestResult, GeneratedOutput, Token, ErrorFlags
from helm.common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
)
from helm.tokenizers.tokenizer import Tokenizer
from helm.clients.client import CachingClient, truncate_sequence


_CONTENT_MODERATION_KEY = "fail.content.moderation.failed"


def _is_content_moderation_failure(response: Dict) -> bool:
    """Return whether a a response failed because of the content moderation filter."""
    errors = response.get("errors")
    if not errors:
        return False
    if len(errors) != 1:
        return False
    return errors[0].get("key") == _CONTENT_MODERATION_KEY


class PalmyraClient(CachingClient):
    def __init__(self, tokenizer: Tokenizer, tokenizer_name: str, cache_config: CacheConfig, api_key: str):
        super().__init__(cache_config=cache_config)
        self.api_key: str = api_key
        self.tokenizer = tokenizer
        self.tokenizer_name = tokenizer_name

    def _send_request(self, model_name: str, raw_request: Dict[str, Any]) -> Dict[str, Any]:
        response = requests.request(
            method="POST",
            url=f"https://enterprise-api.writer.com/llm/organization/3002/model/{model_name}/completions",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            data=json.dumps(raw_request),
        )
        result = json.loads(response.text)
        if "choices" not in result and not _is_content_moderation_failure(result):
            raise ValueError(f"Invalid response: {result}")
        return result

    def make_request(self, request: Request) -> RequestResult:
        """Make a request"""
        raw_request = {
            "prompt": request.prompt,
            "maxTokens": request.max_tokens,
            "temperature": request.temperature,
            "topP": request.top_p,
            "bestOf": request.top_k_per_token,
            "stop": request.stop_sequences,
            # random_seed have been disabled for now.
            # It is here to ensure that Writer does not cache the request when we
            # want several completions with the same prompt. Right now it seems
            # to have no effect so we are disabling it.
            # TODO(#1515): re-enable it when it works.
            # "random_seed": request.random,
        }

        completions: List[GeneratedOutput] = []
        model_name: str = request.model_engine

        # `num_completions` is not supported, so instead make `num_completions` separate requests.
        for completion_index in range(request.num_completions):
            try:

                def do_it() -> Dict[str, Any]:
                    # Add an argument timeout to raw_request to avoid waiting getting timeout of 60s
                    # which happens for long prompts.
                    request_with_timeout = {"timeout": 300, **raw_request}
                    result = self._send_request(model_name, request_with_timeout)
                    return result

                # We need to include the engine's name to differentiate among requests made for different model
                # engines since the engine name is not included in the request itself.
                # In addition, we want to make `request.num_completions` fresh
                # requests, cache key should contain the completion_index.
                # Echoing the original prompt is not officially supported by Writer. We instead prepend the
                # completion with the prompt when `echo_prompt` is true, so keep track of it in the cache key.
                cache_key = CachingClient.make_cache_key(
                    {
                        "engine": request.model_engine,
                        "completion_index": completion_index,
                        **raw_request,
                    },
                    request,
                )

                response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
            except (requests.exceptions.RequestException, AssertionError) as e:
                error: str = f"PalmyraClient error: {e}"
                return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])

            if _is_content_moderation_failure(response):
                hlog(
                    f"WARNING: Returning empty request for {request.model_deployment} "
                    "due to content moderation filter"
                )
                return RequestResult(
                    success=False,
                    cached=False,
                    error=response["errors"][0]["description"],
                    completions=[],
                    embedding=[],
                    error_flags=ErrorFlags(is_retriable=False, is_fatal=False),
                    request_time=response["request_time"],
                    request_datetime=response["request_datetime"],
                )

            response_text: str = response["choices"][0]["text"]

            # The Writer API doesn't support echo. If `echo_prompt` is true, combine the prompt and completion.
            text: str = request.prompt + response_text if request.echo_prompt else response_text
            # The Writer API doesn't return us tokens or logprobs, so we tokenize ourselves.
            tokenization_result: TokenizationRequestResult = self.tokenizer.tokenize(
                # Writer uses the GPT-2 tokenizer
                TokenizationRequest(text, tokenizer=self.tokenizer_name)
            )

            # Log probs are not currently not supported by the Writer, so set to 0 for now.
            tokens: List[Token] = [Token(text=str(text), logprob=0) for text in tokenization_result.raw_tokens]

            completion = GeneratedOutput(text=response_text, logprob=0, tokens=tokens)
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


class PalmyraChatClient(OpenAIClient):
    """Sends request to a Palmyra model using a OpenAI-compatible Chat API."""

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
            base_url="https://api.writer.com/v1/chat",
        )
