from dataclasses import asdict

from icetk import icetk as tokenizer

from helm.common.cache import Cache, CacheConfig
from helm.common.request import Request, RequestResult
from helm.common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
    DecodeRequest,
    DecodeRequestResult,
    TokenizationToken,
)
from .client import Client, wrap_request_time


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
                # TODO: check if tokenize can be replaced with something similar to what we do
                #  in yalm_tokenizer_client.py:
                # token_ids = tokenizer.encode(request.text)
                # tokens = [tokenizer.convert_tokens_to_string([i]) for i in tokenizer.convert_ids_to_tokens(token_ids)]
                # if request.encode:
                #     tokens = token_ids
                # This way we could address #1516 and remove the ▁ in the tests.
                tokens = tokenizer.encode(request.text) if request.encode else tokenizer.tokenize(request.text)
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
