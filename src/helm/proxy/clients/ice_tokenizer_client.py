import os
from dataclasses import asdict

from helm.common.cache import Cache, CacheConfig
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.common.request import Request, RequestResult
from helm.common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
    DecodeRequest,
    DecodeRequestResult,
    TokenizationToken,
)
from .client import Client, wrap_request_time, cleanup_tokens

try:
    # Fall back to pure Python protobufs to work around issue #1613,
    # which is caused by icetk using C++ protobufs compiled with protobuf<3.19.
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    from icetk import icetk as tokenizer
except ModuleNotFoundError as e:
    handle_module_not_found_error(e)


class ICETokenizerClient(Client):
    """
    The ICE Tokenizer is a unified tokenization tool for images, Chinese, and English.
    Source: https://github.com/THUDM/icetk
    """

    def __init__(self, cache_config: CacheConfig):
        self.cache = Cache(cache_config)

    def make_request(self, request: Request) -> RequestResult:
        raise NotImplementedError

    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        cache_key = asdict(request)

        try:

            def do_it():
                tokens = tokenizer.encode(request.text) if request.encode else tokenizer.tokenize(request.text)
                if not request.encode:
                    tokens = cleanup_tokens(tokens, request.tokenizer)
                if request.truncation:
                    tokens = tokens[: request.max_length]
                return {"tokens": tokens}

            result, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except Exception as e:
            error: str = f"ICE Tokenizer error: {e}"
            return TokenizationRequestResult(success=False, cached=False, error=error, text="", tokens=[])

        return TokenizationRequestResult(
            success=True,
            cached=cached,
            text=request.text,
            tokens=[TokenizationToken(value) for value in result["tokens"]],
            request_time=result["request_time"],
        )

    def decode(self, request: DecodeRequest) -> DecodeRequestResult:
        cache_key = asdict(request)

        try:

            def do_it():
                return {"text": tokenizer.decode(request.tokens)}

            result, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except Exception as e:
            error: str = f"ICE Tokenizer error: {e}"
            return DecodeRequestResult(success=False, cached=False, error=error, text="")

        return DecodeRequestResult(
            success=True, cached=cached, text=result["text"], request_time=result["request_time"]
        )
