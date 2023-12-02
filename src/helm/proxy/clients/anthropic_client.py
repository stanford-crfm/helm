from typing import Any, Dict, List, Optional
import json
import requests
import time
import urllib.parse

from helm.common.cache import CacheConfig
from helm.common.hierarchical_logger import htrack_block, hlog
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.common.request import (
    wrap_request_time,
    EMBEDDING_UNAVAILABLE_REQUEST_RESULT,
    Request,
    RequestResult,
    Sequence,
    Token,
    ErrorFlags,
)
from helm.common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
)
from helm.proxy.tokenizers.tokenizer import Tokenizer
from .client import CachingClient, truncate_sequence

try:
    import anthropic
    import websocket
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, ["anthropic"])


class AnthropicClient(CachingClient):
    """
    Client for the Anthropic models (https://arxiv.org/abs/2204.05862).
    They use their own tokenizer.
    Here are a list of bugs that we deal with in this client:
    - The prompt must contains anthropic.HUMAN_PROMPT ('\n\nHuman:') and anthropic.AI_PROMPT ('\n\nAssistant:')
    - The completions is often too verbose, so we add the PROMPT_ANSWER_START to the prompt.
    TODO(#1521): Remove this when we have a better way to do prompt engineering.
    - The completions often start with a colon, space, or newline, so we remove it. This means
    that we need to query more tokens than necessary to ensure that the completion does not start
    with a colon, space, or newline. We query `max_tokens + ADDITIONAL_TOKENS` tokens and then
    remove the excess tokens at the end of the completion.
    TODO(#1512): Once we have a good way to post-process the completion, move this logic to the
    post-processing.
    - The API sometimes returns "Prompt must contain anthropic.AI_PROMPT". This is a bug related to the
    window_service that does not properly truncate some prompts.  It is caused by the suffix being truncated.
    - The API sometimes return Prompt length + max_tokens exceeds max (9192). This is something we do not
    handle yet (we have not limit on prompt length + max_tokens). TODO(#1520): Fix this.
    """

    MAX_COMPLETION_LENGTH: int = (
        8192  # See https://docs.google.com/document/d/1vX6xgoA-KEKxqtMlBVAqYvE8KUfZ7ABCjTxAjf1T5kI/edit#
    )
    ADDITIONAL_TOKENS: int = 5
    PROMPT_ANSWER_START: str = "The answer is "

    def __init__(self, tokenizer: Tokenizer, cache_config: CacheConfig, api_key: Optional[str] = None):
        super().__init__(cache_config=cache_config)
        self.tokenizer = tokenizer
        self.api_key: Optional[str] = api_key
        self._client = anthropic.Client(api_key) if api_key else None

    def _send_request(self, raw_request: Dict[str, Any]) -> Dict[str, Any]:
        if self.api_key is None:
            raise Exception("API key is not set. Please set it in the HELM config file.")
        result = self._client.completion(**raw_request)
        assert "error" not in result, f"Request failed with error: {result['error']}"
        return result

    def _filter_completion(self, completion: str, max_tokens: int) -> str:
        # If the completion starts with a colon, space, or newline, remove it.
        for _ in range(AnthropicClient.ADDITIONAL_TOKENS):
            if len(completion) == 0:
                return completion
            elif completion[0] in [":", " ", "\n"]:
                completion = completion[1:]
            else:
                break

        # NOTE(josselin): For now, this is disabled as it is not an accurate
        # limit of tokens. It is still handled by truncate_sequence() further
        # down the line, but it is not ideal. (prints a warning in the logs)
        # It is now expected that truncate_sequence() will truncate some tokens
        # as we queried more tokens than necessary to ensure that the completion
        # does not start with a colon, space, or newline.

        # Remove excess tokens at the end of the completion.
        # completion = " ".join(completion.split(" ")[:max_tokens])

        return completion

    def make_request(self, request: Request) -> RequestResult:
        if request.max_tokens > AnthropicClient.MAX_COMPLETION_LENGTH:
            raise ValueError(
                "The value for `max_tokens` exceeds the currently supported maximum "
                f"({request.max_tokens} > {AnthropicClient.MAX_COMPLETION_LENGTH})."
            )
        if request.max_tokens == 0 and not request.echo_prompt:
            raise ValueError("echo_prompt must be True when max_tokens=0.")

        raw_request = {
            "prompt": request.prompt,
            "stop_sequences": request.stop_sequences,
            "model": request.model_engine,
            "max_tokens_to_sample": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k_per_token,
        }

        completions: List[Sequence] = []

        # `num_completions` is not supported, so instead make `num_completions` separate requests.
        for completion_index in range(request.num_completions):
            try:

                def do_it():
                    result = self._send_request(raw_request)
                    assert "completion" in result, f"Invalid response: {result}"
                    return result

                # We need to include the engine's name to differentiate among requests made for different model
                # engines since the engine name is not included in the request itself.
                # In addition, we want to make `request.num_completions` fresh
                # requests, cache key should contain the completion_index.
                # Echoing the original prompt is not officially supported by Anthropic. We instead prepend the
                # completion with the prompt when `echo_prompt` is true, so keep track of it in the cache key.
                cache_key = CachingClient.make_cache_key(
                    {
                        "completion_index": completion_index,
                        **raw_request,
                    },
                    request,
                )

                response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
            except Exception as error:
                if "Prompt must contain anthropic.AI_PROMPT" in str(error):
                    return RequestResult(
                        success=False,
                        cached=False,
                        error=str(error),
                        completions=[],
                        embedding=[],
                        error_flags=ErrorFlags(is_retriable=False, is_fatal=False),
                    )
                if "exceeds max (" in str(error):
                    return RequestResult(
                        success=False,
                        cached=False,
                        error=str(error),
                        completions=[],
                        embedding=[],
                        error_flags=ErrorFlags(is_retriable=False, is_fatal=False),
                    )
                return RequestResult(success=False, cached=False, error=str(error), completions=[], embedding=[])

            # Post process the completion.
            response["completion"] = self._filter_completion(response["completion"], request.max_tokens)

            # The Anthropic API doesn't support echo. If `echo_prompt` is true, combine the prompt and completion.
            text: str = request.prompt + response["completion"] if request.echo_prompt else response["completion"]
            # The Anthropic API doesn't return us tokens or logprobs, so we tokenize ourselves.
            tokenization_result: TokenizationRequestResult = self.tokenizer.tokenize(
                # Anthropic uses their own tokenizer
                TokenizationRequest(text, tokenizer=request.model_engine)
            )

            # Log probs are not currently not supported by the Anthropic, so set to 0 for now.
            tokens: List[Token] = [
                Token(text=str(text), logprob=0, top_logprobs={}) for text in tokenization_result.raw_tokens
            ]

            completion = Sequence(text=response["completion"], logprob=0, tokens=tokens)
            # See NOTE() in _filter_completion() to understand why warnings are printed for truncation.
            # TODO(#1512): Fix this with post-processing.
            sequence = truncate_sequence(completion, request, print_warning=True)
            completions.append(sequence)

        return RequestResult(
            success=True,
            cached=cached,
            request_time=response["request_time"],
            request_datetime=response["request_datetime"],
            completions=completions,
            embedding=[],
        )


class AnthropicRequestError(Exception):
    pass


class AnthropicLegacyClient(CachingClient):
    """
    Legacy client for the Anthropic models (https://arxiv.org/abs/2204.05862).
    This was used before they officially released their API on March 17, 2023.
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
            for key in AnthropicLegacyClient.LOGPROBS_RESPONSE_KEYS:
                if key not in response:
                    hlog(f"Invalid logprobs response: {raw_response}. Missing key: {key}")
                    return False
            return True
        except json.decoder.JSONDecodeError:
            hlog(f"Invalid logprobs response: {raw_response}")
            return False

    def __init__(self, api_key: str, tokenizer: Tokenizer, cache_config: CacheConfig):
        hlog("This client is deprecated. Please use AnthropicClient instead.")
        super().__init__(cache_config=cache_config)
        self.api_key = api_key

    def make_request(self, request: Request) -> RequestResult:
        # Embedding not supported for this model
        if request.embedding:
            return EMBEDDING_UNAVAILABLE_REQUEST_RESULT
        # Validate the fields of `Request`
        if request.model_engine != "stanford-online-all-v4-s3":
            raise ValueError(f"Invalid model: {request.model}")
        if request.max_tokens > AnthropicLegacyClient.MAX_COMPLETION_LENGTH:
            raise ValueError(
                "The value for `max_tokens` exceeds the currently supported maximum "
                f"({request.max_tokens} > {AnthropicLegacyClient.MAX_COMPLETION_LENGTH})."
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
                        f"wss://{AnthropicLegacyClient.BASE_ENDPOINT}/model/{request.model_engine}/sample"
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
                            stop_reason = AnthropicLegacyClient.STOP_SEQUENCE_STOP_REASON
                            break

                        tokens.append(token_text)
                        previous_completion_text = completion_text
                    ws.close()
                except websocket.WebSocketException as error:
                    hlog(str(error))
                    raise AnthropicRequestError(f"Anthropic error: {str(error)}")

                # Anthropic doesn't support echoing the prompt, so we have to manually prepend the completion
                # with the prompt when `echo_prompt` is True.
                text: str = request.prompt + response["completion"] if request.echo_prompt else response["completion"]
                logprobs = self.make_logprobs_request(
                    request.prompt + response["completion"], request.top_k_per_token, request.model_engine
                )

                check_logprobs: bool = False
                if not request.echo_prompt:
                    for key in AnthropicLegacyClient.LOGPROBS_RESPONSE_KEYS:
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
                cache_key = CachingClient.make_cache_key(
                    {
                        "engine": request.model_engine,
                        "echo_prompt": request.echo_prompt,
                        "completion_index": completion_index,
                        **raw_request,
                    },
                    request,
                )
                response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
            except AnthropicRequestError as error:
                return RequestResult(success=False, cached=False, error=str(error), completions=[], embedding=[])

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
            if finish_reason == AnthropicLegacyClient.STOP_SEQUENCE_STOP_REASON:
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
            return AnthropicLegacyClient.EMPTY_LOGPROBS_RESPONSE

        raw_response: str

        try:
            logprobs_response = requests.request(
                method="POST",
                url=f"https://{AnthropicLegacyClient.BASE_ENDPOINT}/model/{model_engine}/"
                f"{AnthropicLegacyClient.TOP_K_LOGPROBS_ENDPOINT}",
                headers={
                    "Authorization": f"BEARER {self.api_key}",
                    "Content-Type": "application/json",
                },
                data=json.dumps({"q": text, "k": top_k_per_token, "is_replicated": True}),
            )
            raw_response = logprobs_response.text
        except requests.exceptions.RequestException as error:
            hlog(str(error))
            raise AnthropicRequestError(
                f"Anthropic {AnthropicLegacyClient.TOP_K_LOGPROBS_ENDPOINT} error: {str(error)}"
            )

        if not AnthropicLegacyClient.is_valid_logprobs_response(raw_response):
            raise AnthropicRequestError(f"Invalid logprobs response: {raw_response}")
        return json.loads(raw_response)
