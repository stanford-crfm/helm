from typing import Dict, List, Optional
import json
import time
import urllib.parse

import websocket

from common.cache import Cache
from common.hierarchical_logger import htrack_block, hlog
from common.request import Request, RequestResult, Sequence, Token
from common.tokenization_request import TokenizationRequest, TokenizationRequestResult, TokenizationToken
from .client import Client, wrap_request_time
from .tokenizer.tokenizer_factory import TokenizerFactory


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
    # TODO: increase this later when Anthropic supports more.
    MAX_COMPLETION_LENGTH: int = 3000

    def __init__(self, api_key: str, cache_path: str):
        self.api_key = api_key
        self.cache = Cache(cache_path)
        # Anthropic is using a modified version of the GPT-2 tokenizer.
        self.tokenizer = TokenizerFactory.get_tokenizer("anthropic")

    def make_request(self, request: Request) -> RequestResult:
        # Validate the fields of `Request`
        if request.model != "anthropic/stanford-online-all-v4-s3":
            raise ValueError(f"Invalid model: {request.model}")
        # `Request` field values that Anthropic currently does not support
        if request.echo_prompt:
            raise ValueError("Echoing the original prompt is not supported.")
        if request.top_k_per_token > 1:
            raise ValueError(
                "top_k_per_token > 1 is not supported. The Anthropic API only gives a single token at a time."
            )
        expected_completion_length: int = request.max_tokens - self.tokenizer.tokenize_and_count(request.prompt)
        if expected_completion_length > AnthropicClient.MAX_COMPLETION_LENGTH:
            raise ValueError(
                f"Expected to get back {expected_completion_length} number of tokens in the completion, which "
                f"exceeds the currently supported max completion length of {AnthropicClient.MAX_COMPLETION_LENGTH}."
            )

        raw_request = {
            "q": request.prompt,  # Prompt
            "t": request.temperature,  # Temperature
            # TODO: Recommended to hardcode this to -1 for now
            "k": -1,  # k: ony the top k possibilities
            "p": request.top_p,  # Top p
            "n": request.max_tokens,  # Max tokens
            "stop": request.stop_sequences,  # Stop sequences
            # Anthropic-specific arguments - keep these default values for now.
            "max_simultaneous_queries": 20,  # should be ~20
            # Meta tokens are non-text tokens Anthropic sometimes injects into the text to identify the dataset
            "meta": True,  # meta=True skips sampling meta tokens. Keep it true.
            "is_replicated": True,  # Always set to True
        }

        def do_it():
            with htrack_block("Creating WebSocket connection with Anthropic"):
                try:
                    start: float = time.time()
                    auth: Dict[str, str] = {"key": f"Bearer {self.api_key}"}
                    endpoint: str = (
                        f"wss://feedback-frontend-v2.he.anthropic.com/model/{request.model_engine}/sample"
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
                        token_text: str = completion_text.replace(previous_completion_text, "")
                        tokens.append(token_text)
                        previous_completion_text = completion_text
                    ws.close()
                except websocket.WebSocketException as e:
                    hlog(str(e))
                    raise AnthropicRequestError(f"Anthropic error: {str(e)}")

                # Instead of caching all the responses, just cache what we need
                return {
                    "tokens": tokens,
                    "raw_response": raw_response,
                }

        # Since Anthropic doesn't support multiple completions, we have to manually call it multiple times,
        # and aggregate the results into `completions` and `request_time`.
        completions = []
        all_cached = True
        request_time = 0

        for completion_index in range(request.num_completions):
            try:
                # We need to include the engine's name to differentiate among requests made for different model
                # engines since the engine name is not included in the request itself.
                # In addition, we want to make `request.num_completions` fresh
                # requests, cache key should contain the completion_index.
                cache_key = Client.make_cache_key(
                    {"engine": request.model_engine, "completion_index": completion_index, **raw_request}, request
                )
                response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
            except AnthropicRequestError as e:
                return RequestResult(success=False, cached=False, error=str(e), completions=[])

            token_texts: List[str] = response["tokens"]
            raw_response: Dict = json.loads(response["raw_response"])

            sequence_logprob: float = 0
            tokens: List[Token] = []

            for token_text in token_texts:
                # TODO: Anthropic currently doesn't support logprob. Just set logprob to 0 for now.
                token_logprob: float = 0
                sequence_logprob += token_logprob
                tokens.append(Token(text=token_text, logprob=token_logprob, top_logprobs={}))

            sequence = Sequence(text=raw_response["completion"], logprob=sequence_logprob, tokens=tokens)
            completions.append(sequence)
            request_time += response["request_time"]
            all_cached = all_cached and cached

        return RequestResult(success=True, cached=all_cached, request_time=request_time, completions=completions)

    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        """Tokenizes the text using the underlying tokenizer."""
        return TokenizationRequestResult(
            success=True,
            cached=False,
            tokens=[TokenizationToken(raw_text) for raw_text in self.tokenizer.tokenize(request.text)],
            text=request.text,
        )
