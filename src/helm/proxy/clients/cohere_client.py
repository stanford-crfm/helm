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
from helm.proxy.models import get_models_by_organization
from helm.proxy.tokenizers.tokenizer import Tokenizer
from .client import CachingClient, truncate_sequence
from .cohere_utils import get_cohere_url, DEFAULT_COHERE_API_VERSION


class CohereClient(CachingClient):
    ORGANIZATION: str = "cohere"
    GENERATE_ENDPOINT: str = "generate"

    def __init__(self, api_key: str, tokenizer: Tokenizer, cache_config: CacheConfig):
        super().__init__(cache_config=cache_config, tokenizer=tokenizer)
        self.api_key: str = api_key

    def make_request(self, request: Request) -> RequestResult:
        if request.embedding:
            return EMBEDDING_UNAVAILABLE_REQUEST_RESULT

        # Validate `Request` according to the rules here: https://docs.cohere.ai/generate-reference.
        # Set `return_likelihoods` based on the value of `echo_prompt`:
        # echo_prompt=True => return_likelihoods="ALL"
        # echo_prompt=False => return_likelihoods="GENERATION"
        return_likelihoods: str
        if request.echo_prompt:
            # "If ALL is selected, the token likelihoods will be provided both for the prompt and the generated text."
            return_likelihoods = "ALL"
        else:
            # "If GENERATION is selected, the token likelihoods will only be provided for generated text."
            return_likelihoods = "GENERATION"
            # max_tokens: "Can only be set to 0 if return_likelihoods is set to ALL...",
            # so `max_tokens` has to be greater than 0 when `return_likelihoods` is set to "GENERATION".
            assert request.max_tokens > 0, "max_tokens can only be 0 if echo_prompt=True"

        # model: "Currently available models are small, medium, large, xlarge"
        assert request.model in get_models_by_organization("cohere")
        # temperature: "min value of 0.0, max value of 5.0"
        assert 0.0 <= request.temperature <= 5.0, f"Invalid temperature: {request.temperature}. Valid range: [0,5]"
        # num_generations: "min value of 1, max value of 5"
        assert (
            0 <= request.num_completions <= 5
        ), f"Invalid num_completions: {request.num_completions}. Valid range: [0..5]"
        # k: "Defaults to 0(disabled), which is the minimum. Maximum value is 500"
        assert (
            0 <= request.top_k_per_token <= 500
        ), f"Invalid top_k_per_token: {request.top_k_per_token}. Valid range: [0..500]"
        # p: "Set to 1.0 or 0 to disable. If set to a probability 0.0 < p < 1.0,
        #     it ensures that only the most likely tokens, with total probability mass of p."
        assert 0.0 <= request.top_p <= 1.0, f"Invalid top_p: {request.top_p}. Valid range: [0,1]"
        # frequency_penalty: "min value of 0.0, max value of 1.0"
        assert (
            0.0 <= request.frequency_penalty <= 1.0
        ), f"Invalid frequency_penalty: {request.frequency_penalty}. Valid range: [0,1]"
        # presence_penalty: "min value of 0.0, max value of 1.0"
        assert (
            0.0 <= request.presence_penalty <= 1.0
        ), f"Invalid presence_penalty: {request.presence_penalty}. Valid range: [0,1]"

        raw_request = {
            "prompt": request.prompt,  # Note that "trailing whitespaces [of prompts] will be trimmed".
            "max_tokens": request.max_tokens,
            "model": request.model_engine,
            "temperature": request.temperature,
            "num_generations": request.num_completions,
            "k": request.top_k_per_token,
            "p": request.top_p,
            "frequency_penalty": request.frequency_penalty,
            "presence_penalty": request.presence_penalty,
            "stop_sequences": request.stop_sequences,
            "return_likelihoods": return_likelihoods,
        }

        try:

            def do_it():
                """
                Send the request to the Cohere Generate API. Responses will be structured like this:
                {
                    "generations": [
                        {
                            "text": string,
                            "likelihood": float,
                            "token_likelihoods": [{"token": string, "likelihood": float}, ...]
                        },
                        ...
                    ]
                }

                Note: The stop reason is not included in the response.
                """
                # Cohere has a Python SDK, but it requires additional post-processing to convert their response
                # objects (`Generations`) to JSON, the form the cache expects the responses to be in.
                response = requests.request(
                    method="POST",
                    url=get_cohere_url(CohereClient.GENERATE_ENDPOINT),
                    headers={
                        "Authorization": f"BEARER {self.api_key}",
                        "Content-Type": "application/json",
                        "Cohere-Version": DEFAULT_COHERE_API_VERSION,
                    },
                    data=json.dumps(raw_request),
                )
                result = json.loads(response.text)

                # Error messages are returned through "message"
                assert "message" not in result.keys(), f"Request failed with error {result['message']}"
                assert "generations" in result, f"Invalid response: {result}"
                return result

            response, cached = self.cache.get(raw_request, wrap_request_time(do_it))
        except (requests.exceptions.RequestException, AssertionError) as e:
            error: str = f"CohereClient error: {e}"
            return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])

        completions: List[Sequence] = []
        for generation in response["generations"]:
            # From https://docs.cohere.ai/generate-reference, "the likelihood refers to the average log-likelihood
            # of the entire specified string..." What we want is the sum of the log probabilities of all tokens.
            sequence_logprob: float = 0
            tokens: List[Token] = []
            for token_likelihood in generation["token_likelihoods"]:
                # Cohere does not return the log likelihood for the first token
                # when `echo_prompt=True` or `return_likelihoods` is "ALL".
                logprob: float = token_likelihood.get("likelihood", 0)
                sequence_logprob += logprob

                tokens.append(
                    Token(
                        text=token_likelihood["token"],
                        logprob=logprob,
                        # Cohere does not include the top log probs in the response
                        top_logprobs={},
                    )
                )

            sequence_text: str = generation["text"]
            if request.echo_prompt and request.max_tokens > 0:
                # Cohere does not prepend the original prompt to the output sequence when
                # `return_likelihoods` is "ALL" and `max_tokens` is greater than 0.
                sequence_text = request.prompt + sequence_text

            completion: Sequence = Sequence(text=sequence_text, logprob=sequence_logprob, tokens=tokens)
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
