import json
import requests
from typing import Any, Dict, List

from tokenizers import Tokenizer
from transformers import AutoTokenizer

from helm.common.cache import Cache, CacheConfig
from helm.common.hierarchical_logger import hlog
from helm.common.request import Request, RequestResult, Sequence, Token
from helm.common.tokenization_request import (
    DecodeRequest,
    DecodeRequestResult,
    TokenizationRequest,
    TokenizationRequestResult,
    TokenizationToken,
)
from .client import Client, wrap_request_time, truncate_sequence
from dataclasses import asdict


class PalmyraClient(Client):
    VALID_MODELS: List[str] = ["palmyra-base", "palmyra_large", "palmyra-r"]
    TOKENIZER_NAME_PREFIX = "Writer/"

    @staticmethod
    def _get_model_name(model_name: str) -> str:
        return "/".join(model_name.split("/")[1:])

    def __init__(self, api_key: str, cache_config: CacheConfig):
        self.api_key: str = api_key
        self.cache = Cache(cache_config)
        self._tokenizer_name_to_tokenizer: Dict[str, Tokenizer] = {}

    def _get_tokenizer(self, tokenizer_name: str) -> Tokenizer:
        modified_name = tokenizer_name.replace(self.TOKENIZER_NAME_PREFIX, "")
        if modified_name not in self.VALID_MODELS:
            raise ValueError(f"Invalid tokenizer: {tokenizer_name}")

        # Check if the tokenizer is cached
        if modified_name not in self._tokenizer_name_to_tokenizer:
            self._tokenizer_name_to_tokenizer[modified_name] = AutoTokenizer.from_pretrained(
                self.TOKENIZER_NAME_PREFIX + modified_name, use_fast=False
            )
            hlog(f"Initialized tokenizer: {modified_name}")
        return self._tokenizer_name_to_tokenizer[modified_name]

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
        assert "error" not in result, f"Request failed with error: {result['error']}"
        return result

    def make_request(self, request: Request) -> RequestResult:
        """Make a request"""
        raw_request = {
            "prompt": request.prompt,
            "minTokens": "1",  # Setting to 1 for now
            "maxTokens": request.max_tokens,
            "temperature": request.temperature,
            "topP": request.top_p,
            # bestOf: figure out what this does
        }

        completions: List[Sequence] = []
        model_name: str = request.model_engine

        # `num_completions` is not supported, so instead make `num_completions` separate requests.
        for completion_index in range(request.num_completions):
            try:

                def do_it():
                    result = self._send_request(model_name, raw_request)
                    assert "choices" in result, f"Invalid response: {result}"
                    return result

                # We need to include the engine's name to differentiate among requests made for different model
                # engines since the engine name is not included in the request itself.
                # In addition, we want to make `request.num_completions` fresh
                # requests, cache key should contain the completion_index.
                # Echoing the original prompt is not officially supported by Anthropic. We instead prepend the
                # completion with the prompt when `echo_prompt` is true, so keep track of it in the cache key.
                cache_key = Client.make_cache_key(
                    {
                        "engine": request.model_engine,
                        "echo_prompt": request.echo_prompt,
                        "completion_index": completion_index,
                        **raw_request,
                    },
                    request,
                )

                response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
            except (requests.exceptions.RequestException, AssertionError) as e:
                error: str = f"PalmyraClient error: {e}"
                return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])

            # Post process the completion.
            response_text: str = response["choices"][0]["text"]

            # The Anthropic API doesn't support echo. If `echo_prompt` is true, combine the prompt and completion.
            text: str = request.prompt + response_text if request.echo_prompt else response_text
            # The Anthropic API doesn't return us tokens or logprobs, so we tokenize ourselves.
            tokenization_result: TokenizationRequestResult = self.tokenize(
                # Anthropic uses their own tokenizer
                TokenizationRequest(text, tokenizer="palmyra-base")
            )

            # Log probs are not currently not supported by the Anthropic, so set to 0 for now.
            tokens: List[Token] = [
                Token(text=str(text), logprob=0, top_logprobs={}) for text in tokenization_result.raw_tokens
            ]

            completion = Sequence(text=response_text, logprob=0, tokens=tokens)
            # See NOTE() in _filter_completion() to understand why warnings are disabled.
            sequence = truncate_sequence(completion, request, print_warning=False)
            completions.append(sequence)

        return RequestResult(
            success=True,
            cached=cached,
            request_time=response["request_time"],
            request_datetime=response["request_datetime"],
            completions=completions,
            embedding=[],
        )

    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        cache_key = asdict(request)

        try:

            def do_it():
                tokenizer: Tokenizer = self._get_tokenizer(request.tokenizer)
                tokens = tokenizer.encode(request.text)
                return {"tokens": tokens}

            response, cached = self.cache.get(cache_key, wrap_request_time(do_it))

            return TokenizationRequestResult(
                success=True,
                cached=cached,
                text=request.text,
                tokens=[TokenizationToken(value) for value in response["tokens"]],
                request_time=response["request_time"],
                error=None,
            )
        except Exception as error:
            raise ValueError(f"Palmyra tokenizer error: {error}")

    def decode(self, request: DecodeRequest) -> DecodeRequestResult:
        cache_key = asdict(request)

        try:

            def do_it():
                tokenizer: Tokenizer = self._get_tokenizer(request.tokenizer)
                text: str = tokenizer.decode(request.text)
                return {"text": text}

            response, cached = self.cache.get(cache_key, wrap_request_time(do_it))

            return DecodeRequestResult(
                success=True,
                cached=cached,
                text=str(response["text"]),
                request_time=response["request_time"],
                error=None,
            )
        except Exception as error:
            raise ValueError(f"Palmyra tokenizer error: {error}")
