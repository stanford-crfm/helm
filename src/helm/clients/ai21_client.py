from typing import Dict, List, Optional, TypedDict
import requests

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
from helm.clients.client import CachingClient, truncate_sequence, cleanup_str
from helm.clients.ai21_utils import AI21RequestError, handle_failed_request

try:
    from ai21 import AI21Client as AISDKClient
    from ai21.models.chat import ChatMessage as SDKChatMessage, ChatCompletionResponse
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, ["ai21"])


class AI21Client(CachingClient):
    """
    AI21 Labs provides Jurassic models.
    https://studio.ai21.com/docs/api/
    """

    COMPLETION_URL_TEMPLATE: str = "https://api.ai21.com/studio/v1/{model}/complete"
    EXPERIMENTAL_COMPLETION_URL_TEMPLATE: str = "https://api.ai21.com/studio/v1/experimental/{model}/complete"

    def __init__(self, api_key: str, cache_config: CacheConfig, url: Optional[str] = None):
        super().__init__(cache_config=cache_config)
        self.api_key = api_key
        self.url = url

    def make_request(self, request: Request) -> RequestResult:
        if request.embedding:
            return EMBEDDING_UNAVAILABLE_REQUEST_RESULT

        raw_request = {
            "prompt": request.prompt,
            "temperature": request.temperature,
            "numResults": request.num_completions,
            "topKReturn": request.top_k_per_token,
            "topP": request.top_p,
            "maxTokens": request.max_tokens,
            "stopSequences": request.stop_sequences,
        }

        def do_it():
            if self.url:
                url = self.url
            else:
                url_template: str = (
                    AI21Client.EXPERIMENTAL_COMPLETION_URL_TEMPLATE
                    if request.model_engine == "j1-grande-v2-beta"
                    else AI21Client.COMPLETION_URL_TEMPLATE
                )
                url = url_template.format(model=request.model_engine)
            response = requests.post(
                url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=raw_request,
            ).json()

            # # If 'completions' is not present in the response, assume request failed.
            if "completions" not in response:
                handle_failed_request(api_type="completion", response=response)

            return response

        try:
            # We need to include the engine's name to differentiate among requests made for different model engines
            cache_key = CachingClient.make_cache_key({"engine": request.model_engine, **raw_request}, request)
            response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except AI21RequestError as e:
            return RequestResult(success=False, cached=False, error=str(e), completions=[], embedding=[])

        def fix_text(x: str, first: bool) -> str:
            # TODO(#1522): check if with #1519 this is still needed. This is similar to #1516.
            x = cleanup_str(x, "ai21/j1")
            x = x.replace("<|newline|>", "\n")
            # For some reason, the first token sometimes starts with a space, so get rid of it
            if first and x.startswith(" "):
                x = x[1:]
            return x

        def parse_token(raw: Dict, first: bool) -> Token:
            """
            Parses a raw response token to a Token object.

            Sometimes a "▁" with length 0 is added to the beginning of a sequence
            or token by the AI21 tokenizer probably to mark the start of a new sequence.
            e.g. " burying him" -> ["▁"(0,0), "▁burying"(0,8), "▁him"(8,12)];
            "df\n---" -> '[▁df'(0,2), '\n'(2, 3), '▁---'(3, 6)]

            By computing the actual length of a token and truncating it from the right,
            We can remove those "▁"s so that the tokenization result aligns with the
            input prompt.
            """

            # Compute the actual length of the token text
            # e.g. "▁burying"(0,8) -> 8 - 0 = 8; "▁burying"(0,7) -> 7 - 0 = 7
            text_length: int = raw["textRange"]["end"] - raw["textRange"]["start"]

            return Token(
                # Text should not be longer than text_length. Since "▁" is always inserted
                # in the beginning, we truncate the text from the right.
                text=fix_text(raw["generatedToken"]["token"], first)[-text_length:] if text_length else "",
                logprob=raw["generatedToken"]["raw_logprob"],
            )

        def parse_sequence(raw: Dict, first: bool, finish_reason: Optional[Dict] = None) -> GeneratedOutput:
            text = raw["text"]
            tokens = [parse_token(token, first and i == 0) for i, token in enumerate(raw["tokens"])]
            logprob = sum(token.logprob for token in tokens)
            return GeneratedOutput(text=text, logprob=logprob, tokens=tokens, finish_reason=finish_reason)

        prompt = parse_sequence(response["prompt"], True)
        completions = []
        for raw_completion in response["completions"]:
            completion = parse_sequence(raw_completion["data"], False, raw_completion["finishReason"])
            completion = truncate_sequence(completion, request)
            completions.append(prompt + completion if request.echo_prompt else completion)

        return RequestResult(
            success=True,
            cached=cached,
            request_time=response["request_time"],
            request_datetime=response.get("request_datetime"),
            completions=completions,
            embedding=[],
        )


class AI21ChatRequest(TypedDict):
    """Data passed between make_request and _send_request. Used as the cache key."""

    model: str
    messages: List[Dict[str, str]]
    max_tokens: int
    temperature: float
    stop: List[str]
    n: int
    top_p: float


class AI21ChatClient(CachingClient):
    def __init__(self, api_key: str, cache_config: CacheConfig):
        super().__init__(cache_config=cache_config)
        self.client = AISDKClient(api_key=api_key)

    def make_request(self, request: Request) -> RequestResult:
        if request.embedding:
            return EMBEDDING_UNAVAILABLE_REQUEST_RESULT
        # TODO: Support messages
        assert not request.messages, "AI21ChatClient currently does not support the messages API"

        raw_request: AI21ChatRequest = {
            "model": request.model_engine,
            "messages": [{"role": "user", "content": request.prompt}],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stop": request.stop_sequences,
            "n": request.num_completions,
            "top_p": request.top_p,
        }

        def do_it():
            chat_completion_response: ChatCompletionResponse = self.client.chat.completions.create(
                model=raw_request["model"],
                messages=[SDKChatMessage.from_dict(m) for m in raw_request["messages"]],
                max_tokens=raw_request["max_tokens"],
                temperature=raw_request["temperature"],
                stop=raw_request["stop"],
                n=raw_request["n"],
                top_p=raw_request["top_p"],
            )
            return chat_completion_response.to_dict()

        cache_key = CachingClient.make_cache_key(raw_request, request)
        response, cached = self.cache.get(cache_key, wrap_request_time(do_it))

        completions: List[GeneratedOutput] = []

        for choice in response["choices"]:
            completions.append(GeneratedOutput(text=choice["message"]["content"] or "", logprob=0.0, tokens=[]))

        return RequestResult(
            success=True,
            cached=cached,
            request_time=response["request_time"],
            request_datetime=response["request_datetime"],
            completions=completions,
            embedding=[],
        )
