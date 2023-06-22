from dataclasses import asdict

from helm.common.cache import Cache, CacheConfig
from helm.common.request import (
    Request,
    RequestResult,
    Sequence,
    Token,
    EMBEDDING_UNAVAILABLE_REQUEST_RESULT,
)
from helm.common.tokenization_request import (
    DecodeRequest,
    DecodeRequestResult,
    TokenizationRequest,
    TokenizationRequestResult,
    TokenizationToken,
)
from .client import Client, wrap_request_time

import requests


class NeuripsClient(Client):
    """Implements the client for the NeurIPS LLM Efficiency Challenge."""

    def __init__(self, cache_config: CacheConfig, port: int = 8080):
        self.cache = Cache(cache_config)
        self.port = 8080

    def make_request(self, request: Request) -> RequestResult:
        cache_key = asdict(request)
        # This needs to match whatever we define in pedantic
        if request.embedding:
            return EMBEDDING_UNAVAILABLE_REQUEST_RESULT

        # Only a single stop sequence is supported as we can only pass in a single value for `eos_token_id`
        if len(request.stop_sequences) > 1:
            raise ValueError("More than one stop sequence is not supported.")

        raw_request = {
            "prompt": request.prompt,
            "temperature": 1e-7 if request.temperature == 0 else request.temperature,
            "num_return_sequences": request.num_completions,
            "max_new_tokens": request.max_tokens,
            "top_p": request.top_p,
            "echo_prompt": request.echo_prompt,
            "top_k_per_token": request.top_k_per_token,
            "stop_sequences": request.stop_sequences,
        }

        try:

            def do_it():
                url = f"http://localhost:{self.port}/process"
                response = requests.post(url, json=raw_request)
                response.raise_for_status()
                response_data = response.json()
                return response_data

            response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
            completions = []
            for completion in response["completions"]:
                tokens = [
                    Token(
                        token=token["value"],
                        score=token["score"],
                        start=token["start"],
                        end=token["end"],
                    )
                    for token in completion["tokens"]
                ]
                completions.append(Sequence(tokens=tokens, score=completion["score"]))

            return RequestResult(
                success=True,
                cached=cached,
                error=None,
                completions=completions,
                embedding=[],
            )
        except requests.exceptions.RequestException as e:
            error: str = f"Request error: {e}"
            return RequestResult(
                success=False, cached=False, error=error, completions=[], embedding=[]
            )

    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        cache_key = asdict(request)
        raw_request = {
            "text": request.text,
            "truncation": request.truncation,
            "max_length": request.max_length,
        }

        try:

            def do_it():
                url = f"http://localhost:{self.port}/tokenize"
                response = requests.post(url, json=raw_request)
                response.raise_for_status()
                response_data = response.json()
                return response_data

            result, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except Exception as e:
            error: str = f"Local Model error: {e}"
            return TokenizationRequestResult(
                success=False, cached=False, error=error, text="", tokens=[]
            )

        return TokenizationRequestResult(
            success=True,
            cached=cached,
            text=request.text,
            tokens=[TokenizationToken(value) for value in result["tokens"]],
            request_time=result["request_time"],
        )

    def decode(self, request: DecodeRequest) -> DecodeRequestResult:
        cache_key = asdict(request)

        try:

            def do_it():
                url = f"http://localhost:{self.port}/decode"
                response = requests.post(url, json={"tokens": request.tokens})
                response.raise_for_status()
                response_data = response.json()
                return response_data

            result, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except Exception as e:
            error: str = f"Local Model error: {e}"
            return DecodeRequestResult(success=False, cached=False, error=error, text="")

        return DecodeRequestResult(
            success=True, cached=cached, text=result["text"], request_time=result["request_time"]
        )
