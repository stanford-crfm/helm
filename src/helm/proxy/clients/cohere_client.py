import json
import requests
from typing import List, Dict
from urllib.parse import urljoin

from helm.common.cache import Cache, CacheConfig
from helm.common.request import EMBEDDING_UNAVAILABLE_REQUEST_RESULT, Request, RequestResult, Sequence, Token
from helm.common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
    DecodeRequest,
    DecodeRequestResult,
    TokenizationToken,
)
from helm.proxy.models import get_models_by_organization
from .client import Client, wrap_request_time, truncate_sequence


class CohereClient(Client):
    ORGANIZATION: str = "cohere"

    # From "https://docs.cohere.ai/versioning-reference",
    # "this version [2021-11-08] introduces multiple generations, meaning that the generations endpoint will
    # now accept a num_generations argument in the JSON and will always return an array of generations"
    # Note that the API version is decoupled from the model version.
    DEFAULT_API_VERSION: str = "2021-11-08"

    GENERATE_ENDPOINT: str = "generate"
    TOKENIZE_ENDPOINT: str = "tokenize"

    # According to https://docs.cohere.ai/tokenize-reference#request, for tokenize, text: "the string to
    # be tokenized, the minimum text length is 1 character, and the maximum text length is 65536 characters."
    # However, even sending a request with 60,000 characters sometimes fails, so we set the
    # maximum length to 50,000, which is about 8,333 tokens.
    # TODO: followed up with Cohere support with an example of a failure case
    TOKENIZE_API_MAX_TEXT_LENGTH: int = 50_000

    @staticmethod
    def get_url(endpoint: str) -> str:
        return urljoin("https://api.cohere.ai", endpoint)

    def __init__(self, api_key: str, cache_config: CacheConfig):
        self.api_key: str = api_key
        self.cache = Cache(cache_config)

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
                    url=CohereClient.get_url(CohereClient.GENERATE_ENDPOINT),
                    headers={
                        "Authorization": f"BEARER {self.api_key}",
                        "Content-Type": "application/json",
                        "Cohere-Version": CohereClient.DEFAULT_API_VERSION,
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

    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        text: str = request.text
        assert (
            1 <= len(text) <= CohereClient.TOKENIZE_API_MAX_TEXT_LENGTH
        ), f"Invalid text length: {len(text)}. Valid length: [1..{CohereClient.TOKENIZE_API_MAX_TEXT_LENGTH:,d}]"
        raw_request: Dict[str, str] = {"text": text}

        try:

            def do_it():
                """
                Send the request to the Cohere Tokenize API.

                From https://docs.cohere.ai/tokenize-reference, for text "tokenize me! :D", the response will be:

                {
                    "tokens": [34160, 974, 514, 34, 1420, 69]
                    "token_strings": ["token", "ize", " me", "!", " :", "D"]
                }
                """
                response = requests.request(
                    method="POST",
                    url=CohereClient.get_url(CohereClient.TOKENIZE_ENDPOINT),
                    headers={
                        "Authorization": f"BEARER {self.api_key}",
                        "Content-Type": "application/json",
                        "Cohere-Version": CohereClient.DEFAULT_API_VERSION,
                    },
                    data=json.dumps(raw_request),
                )
                result = json.loads(response.text)
                assert "message" not in result.keys(), f"Request failed with error {result['message']}"
                assert "tokens" in result and "token_strings" in result, f"Invalid response: {result}"
                return result

            response, cached = self.cache.get(raw_request, wrap_request_time(do_it))
        except (requests.exceptions.RequestException, AssertionError) as e:
            error: str = f"CohereClient error: {e}"
            return TokenizationRequestResult(error=error, success=False, cached=False, text="", tokens=[])

        tokens = response["tokens" if request.encode else "token_strings"]
        if request.truncation:
            tokens = tokens[: request.max_length]

        return TokenizationRequestResult(
            success=True,
            cached=cached,
            tokens=[TokenizationToken(value) for value in tokens],
            text=text,
            request_time=response["request_time"],
        )

    def decode(self, request: DecodeRequest) -> DecodeRequestResult:
        raise NotImplementedError("The Cohere API does not support decoding.")
