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
    """Anthropic models. Documentation and paper coming soon."""

    def __init__(self, api_key: str, cache_path: str):
        self.api_key = api_key
        self.cache = Cache(cache_path)
        self.tokenizer = TokenizerFactory.get_tokenizer("anthropic")

    def make_request(self, request: Request) -> RequestResult:
        model: str = request.model
        if model != "anthropic/stanford-online-helpful-v4-s3":
            raise ValueError(f"Invalid model: {model}")

        raw_request = {
            "q": request.prompt,  # Prompt
            "t": request.temperature,  # temperature
            "k": 0,  # TODO: I don't think k is the same as top_k_per_token. Hardcoded to 0 for now.
            "p": request.top_p,  # top p
            "n": request.max_tokens,  # max tokens
            "stop": request.stop_sequences,  # Stop sequences
            # Anthropic-specific arguments - keep these default values for now.
            "max_simultaneous_queries": 20,  # should be ~20
            "meta": True,  # Skip sampling meta tokens. Default to True.
            "is_replicated": True,  # Always set to True
            # Always set to True or it will break for multiple requests with different hyperparameters
            "use_sample_v1": True,
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
                    print(f"Established connection ({websocket_established_connection_time:.2f}s)")

                    # The connection is established. Send the request.
                    ws.send(json.dumps(raw_request))

                    raw_response: str
                    previous_completion_text: str = ""
                    tokens: List[str] = []

                    # Tokens are streamed one at a time. Receive in a loop
                    while True:
                        # 0.4s/tok is pretty standard for Anthropic at the moment for this model size
                        raw_response = ws.recv()

                        if not raw_response:
                            hlog(f"uh-oh...{len(tokens)} tokens in, we are getting an empty response. Trying again...")
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
                    "last_response": raw_response,
                }

        try:
            # We need to include the engine's name to differentiate among requests made for different model
            # engines since the engine name is not included in the request itself.
            cache_key = Client.make_cache_key({"engine": request.model_engine, **raw_request}, request)
            response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except AnthropicRequestError as e:
            return RequestResult(success=False, cached=False, error=str(e), completions=[])

        token_texts: List[str] = response["tokens"]
        last_response: Dict = json.loads(response["last_response"])

        sequence_logprob: float = 0
        tokens: List[Token] = []

        # TODO: handle echo_prompt
        for token_text in token_texts:
            # TODO: Anthropic currently doesn't support logprob
            token_logprob: float = 0
            sequence_logprob += token_logprob
            tokens.append(Token(text=token_text, logprob=token_logprob, top_logprobs={}))
        # TODO: Anthropic currently supports a single completion
        sequence = Sequence(text=last_response["completion"], logprob=sequence_logprob, tokens=tokens)

        return RequestResult(success=True, cached=cached, request_time=response["request_time"], completions=[sequence])

    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        # TODO: add a comment stating that they used a modified version of the GPT-2 tokenizer
        """Tokenizes the text using the GPT-2 tokenizer."""
        return TokenizationRequestResult(
            cached=False, tokens=[TokenizationToken(raw_text) for raw_text in self.tokenizer.tokenize(request.text)]
        )
