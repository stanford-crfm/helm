import json
import requests
from typing import List

from helm.common.cache import CacheConfig
from helm.common.request import (
    wrap_request_time,
    EMBEDDING_UNAVAILABLE_REQUEST_RESULT,
    Request,
    RequestResult,
    Sequence,
    Token,
)
from .client import CachingClient, truncate_sequence
from .cohere_utils import get_cohere_url


class CohereClient(CachingClient):
    """
    Leverages the chat endpoint: https://docs.cohere.com/reference/chat

    Cohere models will only support chat soon: https://docs.cohere.com/docs/migrating-from-cogenerate-to-cochat
    """
    ORGANIZATION: str = "cohere"
    CHAT_ENDPOINT: str = "chat"

    def __init__(self, api_key: str, cache_config: CacheConfig):
        super().__init__(cache_config=cache_config)
        self.api_key: str = api_key

    def _validate_request(self, request: Request):
        assert (
            1 <= request.num_completions <= 1
        ), f"Invalid num_completions: {request.num_completions}. Cohere chat only supports 1 completion at a time."
        assert 0.0 <= request.temperature <= 1.0, f"Invalid temperature: {request.temperature}. Valid range: [0, 1]"
        assert (
            0 <= request.top_k_per_token <= 500
        ), f"Invalid top_k_per_token: {request.top_k_per_token}. Valid range: [0..500]"
        assert 0.0 <= request.top_p <= 1.0, f"Invalid top_p: {request.top_p}. Valid range: [0,1]"
        assert 0.0 <= request.frequency_penalty <= 1.0, f"Invalid frequency_penalty: {request.frequency_penalty}. Valid range: [0,1]"
        assert 0.0 <= request.presence_penalty <= 1.0, f"Invalid presence_penalty: {request.presence_penalty}. Valid range: [0,1]"
        assert 0 <= len(request.stop_sequences) <= 5, f"Invalid length of stop_sequences: {request.stop_sequences}. Up to 5 strings permitted."

    def make_request(self, request: Request) -> RequestResult:
        if request.embedding:
            return EMBEDDING_UNAVAILABLE_REQUEST_RESULT
        self._validate_request(request)
    
        raw_request = {
            "model": request.model_engine,
            "message": request.prompt,
            "max_tokens": request.max_tokens,
            # Setting prompt truncation to off will throw an error if a prompt with too many tokens is passed in
            # This will avoid silent, unexpected behaviour with truncation
            "prompt_truncation": "OFF",
            "temperature": request.temperature,
            "k": request.top_k_per_token,
            "p": request.top_p,
            "frequency_penalty": request.frequency_penalty,
            "presence_penalty": request.presence_penalty,
            "stop_sequences": request.stop_sequences,
        }

        try:

            def do_it():
                """
                Send the request to the Cohere Chat API. Responses will be structured like this:
                cohere.Chat {
                    message: What's up?
                    text: Hey there! How's it going? I'm doing well, thank you for asking 😊.
                    ...
                }
                """
                # Cohere has a Python SDK, but it requires additional post-processing to convert their response
                # objects (`Generations`) to JSON, the form the cache expects the responses to be in.
                response = requests.request(
                    method="POST",
                    url=get_cohere_url(CohereClient.CHAT_ENDPOINT),
                    headers={
                        "Authorization": f"BEARER {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    data=json.dumps(raw_request),
                )
                result = json.loads(response.text)
                assert "text" in result, f"Response does not contain text: {result}"
                return result

            response, cached = self.cache.get(raw_request, wrap_request_time(do_it))
        except (requests.exceptions.RequestException, AssertionError) as e:
            error: str = f"CohereClient error: {e}"
            return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])

        # Cohere chat only supports 1 completion at a time
        # Furthermore, it does not support likelihoods, or return tokens (just text)
        dummy_log_prob = 0.0
        dummy_tokens = []

        completions: List[Sequence] = []
        completion: Sequence = Sequence(text=response["text"], logprob=dummy_log_prob, tokens=dummy_tokens)
        completion = truncate_sequence(completion, request)
        completions.append(completion)

        return RequestResult(
            success=True,
            cached=cached,
            request_time=response["request_time"],
            request_datetime=response["request_datetime"],
            completions=completions,
            embedding=[],
        )
