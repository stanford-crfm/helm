import json
import requests
from typing import Any, Dict, List

from helm.common.cache import Cache, CacheConfig
from helm.common.request import Request, RequestResult, Sequence, Token
from helm.common.tokenization_request import (
    DecodeRequest,
    DecodeRequestResult,
    TokenizationRequest,
    TokenizationRequestResult,
    TokenizationToken,
)
from .client import Client, wrap_request_time, truncate_sequence


class AlephAlphaClient(Client):
    COMPLETION_ENDPOINT: str = "complete"
    TOKENIZE_ENDPOINT: str = "tokenize"
    DETOKENIZE_ENDPOINT: str = "detokenize"

    def __init__(self, api_key: str, cache_config: CacheConfig):
        self.api_key: str = api_key
        self.cache = Cache(cache_config)

    def _send_request(self, endpoint: str, raw_request: Dict[str, Any]) -> Dict[str, Any]:
        response = requests.request(
            method="POST",
            url=f"https://api.aleph-alpha.com/{endpoint}",
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
        """Make a request following https://docs.aleph-alpha.com/api/complete."""
        # TODO: echo is not supported. Follow up on this.
        raw_request = {
            "model": request.model_engine,
            "prompt": request.prompt,
            "maximum_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_k": request.top_k_per_token,
            "top_p": request.top_p,
            "presence_penalty": request.presence_penalty,
            "frequency_penalty": request.frequency_penalty,
            "n": request.num_completions,
            "stop_sequences": request.stop_sequences,
            "log_probs": request.top_k_per_token,
            "tokens": True,  # Setting to True returns individual tokens of the completion
        }

        try:

            def do_it():
                result = self._send_request(AlephAlphaClient.COMPLETION_ENDPOINT, raw_request)
                assert "completions" in result, f"Invalid response: {result}"
                return result

            response, cached = self.cache.get(raw_request, wrap_request_time(do_it))
        except (requests.exceptions.RequestException, AssertionError) as e:
            error: str = f"AlephAlphaClient error: {e}"
            return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])

        completions: List[Sequence] = []
        for completion in response["completions"]:
            sequence_logprob: float = 0
            tokens: List[Token] = []

            # `completion_tokens` is the list of selected tokens.
            for i, token in enumerate(completion["completion_tokens"]):
                # Get the top K logprobs for the ith token
                top_logprobs: Dict[str, float] = completion["log_probs"][i]
                # Use the selected token value to get the logprob
                logprob: float = top_logprobs[token]
                sequence_logprob += logprob
                tokens.append(
                    Token(
                        text=token,
                        logprob=logprob,
                        top_logprobs=top_logprobs,
                    )
                )

            sequence: Sequence = Sequence(text=completion["completion"], logprob=sequence_logprob, tokens=tokens)
            sequence = truncate_sequence(sequence, request)
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
        """Make a request following https://docs.aleph-alpha.com/api/tokenize."""
        raw_request = {
            "model": request.tokenizer_name,
            "prompt": request.text,
            "tokens": True,
            "token_ids": True,
        }

        try:

            def do_it():
                result = self._send_request(AlephAlphaClient.TOKENIZE_ENDPOINT, raw_request)
                assert "tokens" in result and "token_ids" in result, f"Invalid response: {result}"
                return result

            response, cached = self.cache.get(raw_request, wrap_request_time(do_it))
        except (requests.exceptions.RequestException, AssertionError) as e:
            error: str = f"AlephAlphaClient error: {e}"
            return TokenizationRequestResult(error=error, success=False, cached=False, text="", tokens=[])

        tokens = response["token_ids" if request.encode else "tokens"]
        if request.truncation:
            tokens = tokens[: request.max_length]

        return TokenizationRequestResult(
            success=True,
            cached=cached,
            tokens=[TokenizationToken(value) for value in tokens],
            text=request.text,
            request_time=response["request_time"],
        )

    def decode(self, request: DecodeRequest) -> DecodeRequestResult:
        """Make a request following https://docs.aleph-alpha.com/api/detokenize."""
        raw_request = {
            "model": request.tokenizer_name,
            "token_ids": request.tokens,
        }

        try:

            def do_it():
                result = self._send_request(AlephAlphaClient.DETOKENIZE_ENDPOINT, raw_request)
                assert "result" in result, f"Invalid response: {result}"
                return result

            response, cached = self.cache.get(raw_request, wrap_request_time(do_it))
        except (requests.exceptions.RequestException, AssertionError) as e:
            error: str = f"AlephAlphaClient error: {e}"
            return DecodeRequestResult(error=error, success=False, cached=False, text="")

        return DecodeRequestResult(
            success=True,
            cached=cached,
            # The text always seems to start with a single whitespace when encoding/decoding.
            text=response["result"].replace(" ", "", 1),
            request_time=response["request_time"],
        )
