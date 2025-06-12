from abc import abstractmethod
from dataclasses import asdict
from typing import Any, Dict, List

from helm.common.cache import Cache, CacheConfig
from helm.common.request import wrap_request_time
from helm.common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
    DecodeRequest,
    DecodeRequestResult,
    TokenizationToken,
)
from helm.tokenizers.tokenizer import Tokenizer


class CachingTokenizer(Tokenizer):
    def __init__(self, cache_config: CacheConfig) -> None:
        self.cache = Cache(cache_config)

    def _get_tokenizer_name(self, tokenizer: str) -> str:
        return tokenizer.split("/")[-1]

    def _tokenization_request_to_cache_key(self, request: TokenizationRequest) -> Dict[str, Any]:
        """Returns a dictionary that uniquely identifies the tokenization request.
        This is used as the cache key for the tokenization request.

        Most Tokenizer use this simple implementation but it can be overriden
        to implement some custom logic or preserve an existing Cache.
        """
        return asdict(request)

    def _decode_request_to_cache_key(self, request: DecodeRequest) -> Dict[str, Any]:
        """Returns a dictionary that uniquely identifies the decode request.
        This is used as the cache key for the decode request.

        Most Tokenizer use this simple implementation but it can be overriden
        to implement some custom logic or preserve an existing Cache.
        """
        return asdict(request)

    def _tokenization_raw_response_to_tokens(
        self, response: Dict[str, Any], request: TokenizationRequest
    ) -> List[TokenizationToken]:
        """Returns the list of tokens from the raw response.
        This is used to extract the tokens from the raw response.

        Most Tokenizer use this simple implementation but it can be overriden
        to implement some custom logic or preserve an existing Cache.
        """
        return [TokenizationToken(token) for token in response["tokens"]]

    def _decode_raw_response_to_text(self, response: Dict[str, Any], request: DecodeRequest) -> str:
        """Returns the text from the raw response.
        This is used to extract the text from the raw response.

        Most Tokenizer use this simple implementation but it can be overriden
        to implement some custom logic or preserve an existing Cache.
        """
        return response["text"]

    @abstractmethod
    def _tokenize_do_it(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Callable that tokenizes the text and returns a dictionary.
        This dictionnary will then be passed to `_tokenization_raw_response_to_tokens` to extract the tokens.
        The input is a raw request obtained from `_tokenization_request_to_cache_key`.
        """
        pass

    @abstractmethod
    def _decode_do_it(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Callable that decodes the tokens and returns a dictionary.
        This dictionnary will then be passed to `_decode_raw_response_to_text` to extract the text.
        The input is a raw request obtained from `_decode_request_to_cache_key`.
        """
        pass

    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        """Tokenizes `request.text` using `request.tokenizer`.

        This method handles caching and returning the appropriate object while the actual tokenization
        logic lies in the `_get_tokenize_do_it` method. The input for `_get_tokenize_do_it` is a raw
        request obtained from `_get_tokenization_request_to_cache_key`, and the output is post-processed
        by `_post_process_tokenization`.
        Most tokenizers should simply implement the three methods mentionned above and leave this method as is.
        However in some cases, such as the AI21 tokenizer, the tokenization logic is more complex and
        requires additional logic, so this method can be overridden.

        Returns a `TokenizationRequestResult` object.
        """
        raw_request: Dict[str, Any] = self._tokenization_request_to_cache_key(request)

        try:
            # Get the tokens from the cache or compute them
            response, cached = self.cache.get(raw_request, wrap_request_time(lambda: self._tokenize_do_it(raw_request)))
            tokens: List[TokenizationToken] = self._tokenization_raw_response_to_tokens(response, request)
            if request.truncation:
                tokens = tokens[: request.max_length]

            # Internal check of the type of the first token
            # This is to make sure that the tokenization is correct
            if request.encode and len(tokens) > 0:
                assert type(tokens[0].value) == int, (
                    f"tokenize() returned strings instead of integers when encode is True: "
                    f"request={request} repsonse={response}"
                )
            elif not request.encode and len(tokens) > 0:
                assert type(tokens[0].value) == str, (
                    f"tokenize() returned integers instead of strings when encode is False: "
                    f"request={request} repsonse={response}"
                )

            result = TokenizationRequestResult(
                success=True,
                cached=cached,
                text=request.text,
                tokens=tokens,
                request_time=response["request_time"],
                error=None,
            )
            return result
        except Exception as error:
            raise ValueError(f"Failed to tokenize text with {self.__class__.__name__} tokenizer: {error}") from error

    def decode(self, request: DecodeRequest) -> DecodeRequestResult:
        """Decodes `request.tokens` using `request.tokenizer`.

        This method handles caching and returning the appropriate object while the actual decoding
        logic lies in the `_get_decode_do_it` method. The input for `_get_decode_do_it` is a raw
        request obtained from `_get_decode_request_to_cache_key`, and the output is post-processed
        by `_post_process_decode`.
        Most tokenizers hould simply implement the three methods mentionned above and leave this method as is.
        However in some cases, such as the AI21 tokenizer, the decoding logic is more complex and
        requires additional logic, so this method can be overridden.
        """
        raw_request: Dict[str, Any] = self._decode_request_to_cache_key(request)

        try:
            # Get the tokens from the cache or compute them
            response, cached = self.cache.get(raw_request, wrap_request_time(lambda: self._decode_do_it(raw_request)))
            text: str = self._decode_raw_response_to_text(response, request)

            # Internal check of the type of the text
            # This is to make sure that the decoding is correct
            assert type(text) == str

            return DecodeRequestResult(
                success=True,
                cached=cached,
                text=text,
                request_time=response["request_time"],
                error=None,
            )
        except Exception as error:
            raise ValueError(f"Failed to decode tokens with {self.__class__.__name__} tokenizer: {error}") from error
