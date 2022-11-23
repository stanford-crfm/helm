from typing import Any, Dict, List, Optional
import json
import requests
import time
import urllib.parse

import websocket

from helm.common.cache import Cache, CacheConfig
from helm.common.hierarchical_logger import htrack_block, hlog
from helm.common.request import EMBEDDING_UNAVAILABLE_REQUEST_RESULT, Request, RequestResult, Sequence, Token
from helm.common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
    DecodeRequest,
    DecodeRequestResult,
)
from .client import Client, wrap_request_time, truncate_sequence


class AnthropicRequestError(Exception):
    pass


class AnthropicClient(Client):
    """
    Client for the Anthropic models (https://arxiv.org/abs/2204.05862).
    They used their own version of the GPT-2 tokenizer.

    The Anthropic API is not production-ready and currently does not support:
    - Top k per token
    - Multiple completions
    - Echo prompt
    - Log probabilities
    """

    # Note: The model has a maximum context size of 8192, but the Anthropic API
    #       can currently only support a maximum of ~3000 tokens in the completion.
    # TODO: Increase this later when Anthropic supports more.
    MAX_COMPLETION_LENGTH: int = 3000

    # Anthropic returns the following in the response when reaching one of the stop sequences.
    STOP_SEQUENCE_STOP_REASON: str = "stop_sequence"

    ORGANIZATION: str = "anthropic"

    BASE_ENDPOINT: str = "feedback-frontend-v2.he.anthropic.com"
    TOP_K_LOGPROBS_ENDPOINT: str = "topk_logprobs"

    LOGPROBS_RESPONSE_KEYS: List[str] = ["tokens", "logprobs", "topk_tokens", "topk_logprobs"]
    EMPTY_LOGPROBS_RESPONSE: Dict[str, List[Any]] = {
        "tokens": [],
        "logprobs": [],
        "topk_logprobs": [],
        "topk_tokens": [],
    }

    @staticmethod
    def is_valid_logprobs_response(raw_response: str) -> bool:
        try:
            response: Dict = json.loads(raw_response)
            for key in AnthropicClient.LOGPROBS_RESPONSE_KEYS:
                if key not in response:
                    hlog(f"Invalid logprobs response: {raw_response}. Missing key: {key}")
                    return False
            return True
        except json.decoder.JSONDecodeError:
            hlog(f"Invalid logprobs response: {raw_response}")
            return False

    def __init__(self, api_key: str, cache_config: CacheConfig):
        self.api_key = api_key
        self.cache = Cache(cache_config)

    def make_request(self, request: Request) -> RequestResult:
        # Embedding not supported for this model
        if request.embedding:
            return EMBEDDING_UNAVAILABLE_REQUEST_RESULT
        # Validate the fields of `Request`
        if request.model != "anthropic/stanford-online-all-v4-s3":
            raise ValueError(f"Invalid model: {request.model}")
        if request.max_tokens > AnthropicClient.MAX_COMPLETION_LENGTH:
            raise ValueError(
                "The value for `max_tokens` exceeds the currently supported maximum "
                f"({request.max_tokens} > {AnthropicClient.MAX_COMPLETION_LENGTH})."
            )
        if request.max_tokens == 0 and not request.echo_prompt:
            raise ValueError("echo_prompt must be True when max_tokens=0.")

        raw_request = {
            "q": request.prompt,  # Prompt
            "t": request.temperature,  # Temperature
            "k": request.top_k_per_token,  # k: ony the top k possibilities
            "p": request.top_p,  # Top p
            "n": request.max_tokens,  # Max tokens
            # There was a bug recently introduced (07/2022) where the API breaks when a user specifies stop=[]
            # in the request. The workaround is to pass in None instead of an empty list.
            "stop": request.stop_sequences or None,  # Stop sequences.
            # Anthropic-specific arguments - keep these default values for now.
            "max_simultaneous_queries": 20,  # should be ~20
            # Meta tokens are non-text tokens Anthropic sometimes injects into the text to identify the dataset
            "meta": True,  # meta=True skips sampling meta tokens. Keep it true.
            "is_replicated": True,  # Always set to True
        }

        def do_it():
            # Anthropic throws an error when `max_tokens` or `n` is 0, so only send the logprobs request
            if request.max_tokens == 0:
                return {
                    "text": request.prompt,
                    "logprobs": self.make_logprobs_request(
                        request.prompt, request.top_k_per_token, request.model_engine
                    ),
                    "stop_reason": "length",  # Set `stop_reason` to "length" because max_tokens is 0
                }

            with htrack_block("Creating WebSocket connection with Anthropic"):
                try:
                    start: float = time.time()
                    auth: Dict[str, str] = {"key": f"Bearer {self.api_key}"}
                    endpoint: str = (
                        f"wss://{AnthropicClient.BASE_ENDPOINT}/model/{request.model_engine}/sample"
                        f"?{urllib.parse.urlencode(auth)}"
                    )
                    ws = websocket.create_connection(endpoint, header=auth)

                    websocket_established_connection_time: float = time.time() - start
                    hlog(f"Established connection ({websocket_established_connection_time:.2f}s)")

                    # The connection is established. Send the request.
                    ws.send(json.dumps(raw_request))

                    raw_response: str
                    previous_completion_text: str = ""
                    tokens: List[str] = []

                    # Tokens are streamed one at a time. Receive in a loop
                    while True:
                        # 0.4s/tok is pretty standard for Anthropic at the moment for this model size.
                        # If the connection dropped, this throws a `websocket.WebSocketException`.
                        raw_response = ws.recv()

                        if not raw_response:
                            # At this point, if we are getting back an empty response, it's most likely
                            # the connection dropped. We will try again.
                            hlog(f"{len(tokens)} tokens in, but received an empty response. Trying again...")
                            continue

                        response: Dict = json.loads(raw_response)
                        if "exception" in response:
                            raise AnthropicRequestError(f"Anthropic error: {response['exception']}")

                        # Anthropic lets us know when we should stop streaming by sending us a `stop_reason`
                        stop_reason: Optional[str] = response["stop_reason"]
                        # Break out of the loop once we get back a `stop_reason`
                        if stop_reason:
                            hlog(f"Ceasing to send request because of the `stop_reason` in response: {stop_reason}")
                            break

                        completion_text: str = response["completion"]
                        assert completion_text.startswith(previous_completion_text), (
                            f"Could not compute next token:\n"
                            f"request: {raw_request}\n"
                            f"previous: {repr(previous_completion_text)}\n"
                            f"completion: {repr(completion_text)}"
                        )
                        token_text: str = completion_text[len(previous_completion_text) :]
                        # We sometimes get replacement character as the token, but they seem
                        # to disappear in the next iteration, so skip these.
                        if "�" in token_text:
                            hlog(f"Found the replacement character in the token text: {token_text}. Skipping...")
                            continue

                        # Anthropic is sending us excess tokens beyond the stop sequences,
                        # so we have to stop early ourselves.
                        if any(stop in token_text for stop in request.stop_sequences):
                            hlog(f"Received {repr(token_text)}, which has a stop sequence - early stopping.")
                            stop_reason = AnthropicClient.STOP_SEQUENCE_STOP_REASON
                            break

                        tokens.append(token_text)
                        previous_completion_text = completion_text
                    ws.close()
                except websocket.WebSocketException as e:
                    hlog(str(e))
                    raise AnthropicRequestError(f"Anthropic error: {str(e)}")

                # Anthropic doesn't support echoing the prompt, so we have to manually prepend the completion
                # with the prompt when `echo_prompt` is True.
                text: str = request.prompt + response["completion"] if request.echo_prompt else response["completion"]
                logprobs = self.make_logprobs_request(
                    request.prompt + response["completion"], request.top_k_per_token, request.model_engine
                )

                check_logprobs: bool = False
                if not request.echo_prompt:
                    for key in AnthropicClient.LOGPROBS_RESPONSE_KEYS:
                        # This is a naive approach where we just take the last k tokens and log probs,
                        # where k is the number of tokens in the completion. Ideally, log probs would
                        # be included as part of the response for the inference endpoint.
                        logprobs[key] = logprobs[key][-len(tokens) :]

                    if logprobs["tokens"] != tokens:
                        # This is a known limitation with the Anthropic API. For now keep track of the
                        # entries with the mismatch.
                        hlog(
                            f"WARNING: naive truncation for logprobs did not work."
                            f"\nRequest:{raw_request}\nExpected: {tokens}\nActual: {logprobs['tokens']}"
                        )
                        check_logprobs = True

                return {
                    "text": text,
                    "logprobs": logprobs,
                    "stop_reason": stop_reason,
                    "check_logprobs": check_logprobs,
                }

        # Since Anthropic doesn't support multiple completions, we have to manually call it multiple times,
        # and aggregate the results into `completions` and `request_time`.
        completions: List[Sequence] = []
        all_cached = True
        request_time = 0
        request_datetime: Optional[int] = None

        for completion_index in range(request.num_completions):
            try:
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
            except AnthropicRequestError as e:
                return RequestResult(success=False, cached=False, error=str(e), completions=[], embedding=[])

            sequence_logprob: float = 0
            tokens: List[Token] = []
            log_probs: Dict[str, List[Any]] = response["logprobs"]

            for text, token_logprob, all_logprobs, all_tokens in zip(
                log_probs["tokens"], log_probs["logprobs"], log_probs["topk_logprobs"], log_probs["topk_tokens"]
            ):
                top_logprobs: Dict[str, float] = {text: logprob for text, logprob in zip(all_tokens, all_logprobs)}
                tokens.append(Token(text=text, logprob=token_logprob, top_logprobs=top_logprobs))
                sequence_logprob += token_logprob

            finish_reason: str = response["stop_reason"]
            # Maintain uniformity with other APIs
            if finish_reason == AnthropicClient.STOP_SEQUENCE_STOP_REASON:
                finish_reason = "stop"

            completion = Sequence(
                text=response["text"],
                logprob=sequence_logprob,
                tokens=tokens,
                finish_reason={"reason": finish_reason},
            )
            completion = truncate_sequence(completion, request)
            completions.append(completion)
            request_time += response["request_time"]
            # Use the datetime from the first completion because that's when the request was fired
            request_datetime = request_datetime or response.get("request_datetime")
            all_cached = all_cached and cached

        return RequestResult(
            success=True,
            cached=all_cached,
            request_time=request_time,
            request_datetime=request_datetime,
            completions=completions,
            embedding=[],
        )

    def make_logprobs_request(self, text: str, top_k_per_token: int, model_engine: str) -> Dict[str, List[Any]]:
        """
        Get the token log probs and top candidates for a given text using the endpoint: topk_logprobs.
        """
        # Sending an empty string results in 'non cancel Cannot evaluate top logprobs of empty string' error
        if len(text) == 0:
            return AnthropicClient.EMPTY_LOGPROBS_RESPONSE

        raw_response: str

        try:
            logprobs_response = requests.request(
                method="POST",
                url=f"https://{AnthropicClient.BASE_ENDPOINT}/model/{model_engine}/"
                f"{AnthropicClient.TOP_K_LOGPROBS_ENDPOINT}",
                headers={
                    "Authorization": f"BEARER {self.api_key}",
                    "Content-Type": "application/json",
                },
                data=json.dumps({"q": text, "k": top_k_per_token, "is_replicated": True}),
            )
            raw_response = logprobs_response.text
        except requests.exceptions.RequestException as e:
            hlog(str(e))
            raise AnthropicRequestError(f"Anthropic {AnthropicClient.TOP_K_LOGPROBS_ENDPOINT} error: {str(e)}")

        if not AnthropicClient.is_valid_logprobs_response(raw_response):
            raise AnthropicRequestError(f"Invalid logprobs response: {raw_response}")
        return json.loads(raw_response)

    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        raise NotImplementedError("Use the HuggingFaceClient to tokenize.")

    def decode(self, request: DecodeRequest) -> DecodeRequestResult:
        raise NotImplementedError("Use the HuggingFaceClient to decode.")
