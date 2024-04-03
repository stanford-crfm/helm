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
    ORGANIZATION: str = "cohere"
    CHAT_ENDPOINT: str = "chat"

    def __init__(self, api_key: str, cache_config: CacheConfig):
        super().__init__(cache_config=cache_config)
        self.api_key: str = api_key

    def make_request(self, request: Request) -> RequestResult:
        if request.embedding:
            return EMBEDDING_UNAVAILABLE_REQUEST_RESULT

        # Validate `Request` according to the rules here: https://docs.cohere.com/reference/chat.

        assert 0.0 <= request.temperature <= 1.0, f"Invalid temperature: {request.temperature}. Valid range: [0, 1]"

        assert (
            1 <= request.num_completions <= 1
        ), f"Invalid num_completions: {request.num_completions}. Only 1 can be requested at a time."

        assert (
            0 <= request.top_k_per_token <= 500
        ), f"Invalid top_k_per_token: {request.top_k_per_token}. Valid range: [0..500]"

        assert 0.0 <= request.top_p <= 1.0, f"Invalid top_p: {request.top_p}. Valid range: [0,1]"
    
        raw_request = {
            "model": request.model_engine,
            "message": request.prompt,
            "max_tokens": request.max_tokens,
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
                Send the request to the Cohere Generate API. Responses will be structured like this:
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
                assert "text" in result, f"Invalid response: {result}"
                return result

            response, cached = self.cache.get(raw_request, wrap_request_time(do_it))
        except (requests.exceptions.RequestException, AssertionError) as e:
            error: str = f"CohereClient error: {e}"
            return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])

        completions: List[Sequence] = []
        completion: Sequence = Sequence(text=response["text"], logprob=1.0, tokens=[])
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
