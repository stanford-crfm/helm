import json
import requests
from typing import Any, Dict, List
import traceback
from helm.common.cache import CacheConfig

from helm.common.request import (
    wrap_request_time,
    EMBEDDING_UNAVAILABLE_REQUEST_RESULT,
    Request,
    RequestResult,
    GeneratedOutput,
    Token,
)
from helm.common.tokenization_request import TokenizationRequest
from helm.clients.client import CachingClient, truncate_sequence
from helm.tokenizers.tokenizer import Tokenizer


class MegatronClient(CachingClient):
    """Client for remote Megatron-LM server.

    This client expects an external Megatron-LM server to be run on localhost:5000. See the
    Megatron-LM respository for documentation on starting a Megatron text generation server:

    https://github.com/NVIDIA/Megatron-LM#gpt-text-generation
    """

    def __init__(self, tokenizer: Tokenizer, tokenizer_name: str, cache_config: CacheConfig):
        super().__init__(cache_config=cache_config)
        self.tokenizer = tokenizer
        self.tokenizer_name = tokenizer_name

    def _send_request(self, raw_request: Dict[str, Any]) -> Dict[str, Any]:
        response = requests.request(
            method="PUT",
            # TODO(tgale): Make this configurable.
            url="http://localhost:5000/api",
            headers={
                "Content-Type": "application/json; charset=UTF-8",
            },
            data=json.dumps(raw_request),
        )
        out = json.loads(response.text)

        # Detect if the server returned an error string.
        if type(out) != dict:
            raise ValueError(f"{response}: {response.text}")
        return out

    def _tokenize_response(self, text: str) -> List[Token]:
        tokenized_text = self.tokenizer.tokenize(TokenizationRequest(text, tokenizer=self.tokenizer_name))

        # TODO(tgale): Support logprobs.
        tokens = [Token(text=str(token), logprob=0) for token in tokenized_text.raw_tokens]
        return tokens

    def _make_request(self, request: Request) -> RequestResult:
        # Embedding not supported for this model
        if request.embedding:
            return EMBEDDING_UNAVAILABLE_REQUEST_RESULT

        # TODO(tgale): Relax these.
        assert request.num_completions == 1
        assert not request.echo_prompt
        assert not request.stop_sequences
        assert request.top_p == 1

        # TODO(tgale): Handle log probabilities.
        raw_request = {
            "prompts": [request.prompt],
            "tokens_to_generate": request.max_tokens,
            "temperature": request.temperature,
            "top_k": request.top_k_per_token,
        }

        cache_key = CachingClient.make_cache_key(raw_request, request)
        response, cached = self.cache.get(cache_key, wrap_request_time(lambda: self._send_request(raw_request)))

        # Verify we got a single response for the prompt.
        assert len(response["text"]) == 1

        # NOTE: Megatron returns the response with the prompt included.
        generated_text = response["text"][0]
        if not request.echo_prompt:
            generated_text = generated_text[len(request.prompt) :]

        # NOTE: Megatron returns the de-tokenized response. Re-tokenize.
        tokens = self._tokenize_response(generated_text)
        completion = GeneratedOutput(text=generated_text, logprob=0, tokens=tokens)
        completion = truncate_sequence(completion, request, print_warning=True)

        return RequestResult(
            success=True,
            cached=cached,
            request_time=response["request_time"],
            request_datetime=response.get("request_datetime"),
            completions=[completion],
            embedding=[],
        )

    def make_request(self, request: Request) -> RequestResult:
        try:
            return self._make_request(request)
        except Exception as e:
            return RequestResult(
                success=False,
                cached=False,
                error=f"MegatronClient Error: {e}\n\n{traceback.format_exc()}",
                completions=[],
                embedding=[],
            )
