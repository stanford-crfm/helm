from dataclasses import asdict

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
from .yalm_tokenizer.yalm_tokenizer import YaLMTokenizer


class YaLMTokenizerClient(Client):
    """
    The tokenizer for YaLM, which was trained on Russian and English text.
    Source: https://github.com/yandex/YaLM-100B
    """

    def __init__(self, cache_config: CacheConfig):
        self.cache = Cache(cache_config)
        self.tokenizer = YaLMTokenizer()

    def make_request(self, request: Request) -> RequestResult:
        raise NotImplementedError

    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        cache_key = asdict(request)

        try:

            def do_it():
                token_ids = self.tokenizer.tokenize(request.text)
                if request.truncation:
                    token_ids = token_ids[: request.max_length]
                return {"tokens": token_ids if request.encode else self.tokenizer.convert_ids_to_tokens(token_ids)}
                # TODO(1522): Check if we can reenable this to remove "▁"
                # return {"tokens": token_ids if request.encode else self.tokenizer.convert_ids_to_string(token_ids)}

            result, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except Exception as e:
            error: str = f"YaLM Tokenizer error: {e}"
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
                return {"text": self.tokenizer.decode(request.tokens)}

            result, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except Exception as e:
            error: str = f"YaLM Tokenizer error: {e}"
            return DecodeRequestResult(success=False, cached=False, error=error, text="")

        return DecodeRequestResult(
            success=True, cached=cached, text=result["text"], request_time=result["request_time"]
        )
