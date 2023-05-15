import time
from typing import Any, Dict, List, Optional

from filelock import FileLock

from helm.common.cache import Cache, CacheConfig
from helm.common.hierarchical_logger import hlog
from helm.common.request import Request, RequestResult, Sequence, Token
from helm.common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
    DecodeRequest,
    DecodeRequestResult,
)
from .client import Client, truncate_sequence, wrap_request_time


class ChatGPTClient(Client):
    """
    Client for OpenAI's ChatGPT (https://openai.com/blog/chatgpt).
    We use the unofficial ChatGPT Python client: https://github.com/acheong08/ChatGPT.
    See https://github.com/acheong08/ChatGPT/wiki/Setup for the setup. The client requires Chrome or Chromium installed.
    """

    REQUEST_TIMEOUT_SECONDS: int = 10 * 60  # 10 minutes
    SECONDS_BETWEEN_REQUESTS: int = 1 * 60  # 1 minute

    def __init__(self, session_token: str, lock_file_path: str, cache_config: CacheConfig, tokenizer_client: Client):
        self.session_token: str = session_token
        # Initialize `Chatbot` when we're ready to make the request
        self.chat_bot = None
        self.tokenizer_client: Client = tokenizer_client
        self.cache = Cache(cache_config)

        # Since we want a brand new chat session per request, only allow a single request at a time.
        self._lock = FileLock(lock_file_path, timeout=ChatGPTClient.REQUEST_TIMEOUT_SECONDS)

    def _get_chat_bot_client(self):
        # Import when needed. This library breaks GHA pipeline.
        from revChatGPT.ChatGPT import Chatbot

        if self.chat_bot is None:
            self.chat_bot = Chatbot({"session_token": self.session_token})
        return self.chat_bot

    def make_request(self, request: Request) -> RequestResult:
        completions: List[Sequence] = []
        all_cached = True
        request_time = 0
        request_datetime: Optional[int] = None

        # `num_completions` is not supported, so instead make `num_completions` separate requests.
        for completion_index in range(request.num_completions):
            try:

                def do_it():
                    with self._lock:
                        chat_bot = self._get_chat_bot_client()
                        chat_bot.refresh_session()
                        result: Dict[str, Any] = chat_bot.ask(request.prompt)
                        assert "message" in result, f"Invalid response: {result}"
                        hlog(f"Response: {result['message']}")
                        chat_bot.reset_chat()
                        return result

                raw_request: Dict[str, Any] = {
                    "model": request.model_engine,
                    "prompt": request.prompt,
                    "completion_index": completion_index,
                }
                cache_key = Client.make_cache_key(raw_request, request)
                response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
                if not cached:
                    # Add some sleep between requests to attempt to hit the rate limit.
                    # The rate limit seems to be ~60 requests/hour
                    hlog("Made a request. Sleeping...")
                    time.sleep(ChatGPTClient.SECONDS_BETWEEN_REQUESTS)
            except Exception as e:
                error: str = f"ChatGPT error: {e}"
                return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])

            # The ChatGPT API doesn't support echo. If `echo_prompt` is true, combine the prompt and completion.
            text: str = request.prompt + response["message"] if request.echo_prompt else response["message"]
            # The ChatGPT API doesn't return us tokens or logprobs, so we tokenize ourselves.
            tokenization_result: TokenizationRequestResult = self.tokenize(
                # We're assuming ChatGPT uses the GPT-2 tokenizer.
                TokenizationRequest(text, tokenizer="huggingface/gpt2")
            )

            # Log probs are not currently not supported by the ChatGPT, so set to 0 for now.
            tokens: List[Token] = [
                Token(text=str(text), logprob=0, top_logprobs={}) for text in tokenization_result.raw_tokens
            ]
            completion = Sequence(text=response["message"], logprob=0, tokens=tokens)
            completions.append(truncate_sequence(completion, request))  # Truncate the text by stop sequences
            request_time += response["request_time"]
            request_datetime = request_datetime or response["request_datetime"]
            all_cached = all_cached and cached

        return RequestResult(
            success=True,
            cached=all_cached,
            request_time=request_time,
            request_datetime=request_datetime,
            completions=completions,
            embedding=[],
        )

    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        return self.tokenizer_client.tokenize(request)

    def decode(self, request: DecodeRequest) -> DecodeRequestResult:
        return self.tokenizer_client.decode(request)
