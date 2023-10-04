from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Callable, Any, Dict, List, Optional

from helm.common.cache import Cache, CacheConfig
from helm.common.request import wrap_request_time
from helm.common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
    DecodeRequest,
    DecodeRequestResult,
    TokenizationToken,
)


class Tokenizer(ABC):
    def __init__(self, cache_config: CacheConfig) -> None:
        self.cache = Cache(cache_config)

    def _check_tokenizer_is_supported(self, tokenizer_name: str) -> None:
        """Checks that `tokenizer_name` is supported by this tokenizer."""
        if tokenizer_name not in self.supported_tokenizers:
            raise ValueError(
                f"{tokenizer_name} is not supported. Supported tokenizers are: {self.supported_tokenizers}"
            )

    def _get_tokenizer_name(self, tokenizer: str) -> str:
        return tokenizer.split("/")[-1]

    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        """Tokenizes `request.text` using `request.tokenizer`.

        This method handles caching and returning the appropriate object while the actual tokenization
        logic lies in the `_get_tokenize_do_it` method.
        Most tokenizers hould simply implement `_get_tokenize_do_it` and leave this method as is.
        However in some cases, such as the AI21 tokenizer, the tokenization logic is more complex and
        requires additional logic, so this method can be overridden.

        Returns a `TokenizationRequestResult` object.
        """
        self._check_tokenizer_is_supported(request.tokenizer)
        cache_key = asdict(request)

        try:
            response, cached = self.cache.get(cache_key, wrap_request_time(self._get_tokenize_do_it(request)))

            result = TokenizationRequestResult(
                success=True,
                cached=cached,
                text=request.text,
                tokens=[TokenizationToken(value) for value in response["tokens"]],
                request_time=response["request_time"],
                error=None,
            )
            return result
        except Exception as error:
            raise ValueError(f"Failed to tokenize text with {self.__class__.__name__} tokenizer: {error}") from error

    def decode(self, request: DecodeRequest) -> DecodeRequestResult:
        """Decodes `request.tokens` using `request.tokenizer`.

        This method handles caching and returning the appropriate object while the actual decoding
        logic lies in the `_get_decode_do_it` method.
        Most tokenizers hould simply implement `_get_decode_do_it` and leave this method as is.
        However in some cases, such as the AI21 tokenizer, the decoding logic is more complex and
        requires additional logic, so this method can be overridden.
        """
        self._check_tokenizer_is_supported(request.tokenizer)
        cache_key = asdict(request)

        try:
            response, cached = self.cache.get(cache_key, wrap_request_time(self._get_decode_do_it(request)))

            return DecodeRequestResult(
                success=True,
                cached=cached,
                text=str(response["text"]),
                request_time=response["request_time"],
                error=None,
            )
        except Exception as error:
            raise ValueError(f"Failed to decode tokens with {self.__class__.__name__} tokenizer: {error}") from error

    @abstractmethod
    def _get_tokenize_do_it(self, request: TokenizationRequest) -> Callable[[], Dict[str, Any]]:
        """Returns a callable that tokenizes the text and returns a dictionary with the expected key:
            - "tokens": a list of tokens
        Additional keys can be added if a custom `tokenize` method is implemented. Otherwise, the
        default implementation will ignore them.
        """
        pass

    @abstractmethod
    def _get_decode_do_it(self, request: DecodeRequest) -> Callable[[], Dict[str, Any]]:
        """Returns a callable that decodes the tokens and returns a dictionary with the expected key:
            - "text": the decoded text
        Additional keys can be added if a custom `decode` method is implemented. Otherwise, the
        default implementation will ignore them.
        """
        pass

    @property
    @abstractmethod
    def supported_tokenizers(self) -> List[str]:
        """Returns a list of supported tokenizers."""
        pass


def cleanup_str(token: str, tokenizer_name: Optional[str] = None) -> str:
    """
    Certain tokenizers introduce special characters to represent spaces, such as
    "Ġ" or "▁". This function removes those characters.
    """
    if tokenizer_name in [
        "TsinghuaKEG/ice",
        "bigscience/T0pp",
        "google/t5-11b",
        "google/flan-t5-xxl",
        "google/ul2",
        "Yandex/yalm",
        "ai21/j1",
        "together",
    ]:
        return token.replace("▁", " ")
    elif tokenizer_name is not None and tokenizer_name.startswith("huggingface"):
        return token.replace("Ġ", " ")
    return token


def cleanup_tokens(tokens: List[str], tokenizer_name: Optional[str] = None) -> List[str]:
    """
    Applies `cleanup_str` to each token in `tokens`.
    """
    return [cleanup_str(token, tokenizer_name) for token in tokens]
