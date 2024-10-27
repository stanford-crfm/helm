import json
import requests
from typing import List, Optional, Sequence, TypedDict

from helm.common.cache import CacheConfig
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.common.request import (
    wrap_request_time,
    EMBEDDING_UNAVAILABLE_REQUEST_RESULT,
    Request,
    RequestResult,
    GeneratedOutput,
    Token,
)
from helm.clients.client import CachingClient, truncate_sequence
from helm.clients.cohere_utils import get_cohere_url, DEFAULT_COHERE_API_VERSION

try:
    import cohere
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, ["cohere"])


class CohereClient(CachingClient):
    ORGANIZATION: str = "cohere"
    GENERATE_ENDPOINT: str = "generate"

    def __init__(self, api_key: str, cache_config: CacheConfig):
        super().__init__(cache_config=cache_config)
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

        completions: List[GeneratedOutput] = []
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

                tokens.append(Token(text=token_likelihood["token"], logprob=logprob))

            sequence_text: str = generation["text"]
            if request.echo_prompt and request.max_tokens > 0:
                # Cohere does not prepend the original prompt to the output sequence when
                # `return_likelihoods` is "ALL" and `max_tokens` is greater than 0.
                sequence_text = request.prompt + sequence_text

            completion: GeneratedOutput = GeneratedOutput(text=sequence_text, logprob=sequence_logprob, tokens=tokens)
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


class CohereRawChatRequest(TypedDict):
    message: str
    model: Optional[str]
    preamble: Optional[str]
    chat_history: Optional[Sequence[cohere.ChatbotMessage]]
    temperature: Optional[float]
    max_tokens: Optional[int]
    k: Optional[int]
    p: Optional[float]
    seed: Optional[int]
    stop_sequences: Optional[Sequence[str]]
    frequency_penalty: Optional[float]
    presence_penalty: Optional[float]


def convert_to_raw_chat_request(request: Request) -> CohereRawChatRequest:
    # TODO: Support chat
    model = request.model.replace("cohere/", "")
    return {
        "message": request.prompt,
        "model": model,
        "preamble": None,
        "chat_history": None,
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
        "k": request.top_k_per_token,
        "p": request.top_p,
        "stop_sequences": request.stop_sequences,
        "seed": int(request.random) if request.random is not None else None,
        "frequency_penalty": request.frequency_penalty,
        "presence_penalty": request.presence_penalty,
    }


class CohereChatClient(CachingClient):
    """
    Leverages the chat endpoint: https://docs.cohere.com/reference/chat

    Cohere models will only support chat soon: https://docs.cohere.com/docs/migrating-from-cogenerate-to-cochat
    """

    def __init__(self, api_key: str, cache_config: CacheConfig):
        super().__init__(cache_config=cache_config)
        self.client = cohere.Client(api_key=api_key)

    def make_request(self, request: Request) -> RequestResult:
        if request.embedding:
            return EMBEDDING_UNAVAILABLE_REQUEST_RESULT
        # TODO: Support multiple completions
        assert request.num_completions == 1, "CohereChatClient only supports num_completions=1"
        # TODO: Support messages
        assert not request.messages, "CohereChatClient currently does not support the messages API"

        raw_request: CohereRawChatRequest = convert_to_raw_chat_request(request)

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
                raw_response = self.client.chat(**raw_request).dict()
                assert "text" in raw_response, f"Response does not contain text: {raw_response}"
                return raw_response

            response, cached = self.cache.get(raw_request, wrap_request_time(do_it))
        except (requests.exceptions.RequestException, AssertionError) as e:
            error: str = f"CohereClient error: {e}"
            return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])

        completions: List[GeneratedOutput] = []
        completion: GeneratedOutput = GeneratedOutput(text=response["text"], logprob=0.0, tokens=[])
        completions.append(completion)

        return RequestResult(
            success=True,
            cached=cached,
            request_time=response["request_time"],
            request_datetime=response["request_datetime"],
            completions=completions,
            embedding=[],
        )
