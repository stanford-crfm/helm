from abc import abstractmethod
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from helm.common.cache import Cache, CacheConfig
from helm.common.request import wrap_request_time
from helm.common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
    DecodeRequest,
    DecodeRequestResult,
    TokenizationToken,
)
from .tokenizer import Tokenizer


class CachableTokenizer(Tokenizer):
    def __init__(self, cache_config: CacheConfig) -> None:
        self.cache = Cache(cache_config)

    def _get_tokenizer_name(self, tokenizer: str) -> str:
        return tokenizer.split("/")[-1]

    @property
    def use_encode_in_cache_key(self) -> bool:
        """Whether to use the `encode` parameter in the cache key.

        This is a small optimization as some tokenizers directly compute both
        the token strings and the token ids, so the `encode` parameter is not
        used in the tokenization logic.
        """
        return True

    def _post_process_tokenization(
        self, tokens: List[TokenizationToken], request: TokenizationRequest
    ) -> List[TokenizationToken]:
        """Post-processes the tokens before returning them.

        This method is called after tokenization and after caching.
        It is useful to make this class more modular as some tokenizers need
        special post-processing.
        """
        return tokens

    def _post_process_decoding(self, text: str, request: DecodeRequest) -> str:
        """Post-processes the decoded text before returning it.

        This method is called after decoding and after caching.
        It is useful to make this class more modular as some tokenizers need
        special post-processing.
        """
        return text

    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        """Tokenizes `request.text` using `request.tokenizer`.

        This method handles caching and returning the appropriate object while the actual tokenization
        logic lies in the `_get_tokenize_do_it` method.
        Most tokenizers hould simply implement `_get_tokenize_do_it` and leave this method as is.
        However in some cases, such as the AI21 tokenizer, the tokenization logic is more complex and
        requires additional logic, so this method can be overridden.

        Returns a `TokenizationRequestResult` object.
        """
        cache_key = {
            "text": request.text,
            "tokenizer": request.tokenizer,
            "max_length": request.max_length if request.truncation else None,
            "encode": request.encode if self.use_encode_in_cache_key else None,
        }

        try:

            def do_it():
                response = self._tokenize_do_it(request)
                if request.truncation:
                    if "token_ids" in response:
                        response["token_ids"] = response["token_ids"][: request.max_length]
                    if "token_strings" in response:
                        response["token_strings"] = response["token_strings"][: request.max_length]
                return response

            response, cached = self.cache.get(cache_key, wrap_request_time(do_it))

            if request.encode:
                assert "token_ids" in response
            else:
                assert "token_strings" in response

            # Post process tokens
            token_values = response["token_ids"] if request.encode else response["token_strings"]
            tokens = [TokenizationToken(value) for value in token_values]
            tokens = self._post_process_tokenization(tokens, request)

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
        logic lies in the `_get_decode_do_it` method.
        Most tokenizers hould simply implement `_get_decode_do_it` and leave this method as is.
        However in some cases, such as the AI21 tokenizer, the decoding logic is more complex and
        requires additional logic, so this method can be overridden.
        """
        cache_key = asdict(request)

        try:
            response, cached = self.cache.get(cache_key, wrap_request_time(lambda: self._decode_do_it(request)))

            # Post process text
            text = str(response["text"])
            text = self._post_process_decoding(text, request)

            return DecodeRequestResult(
                success=True,
                cached=cached,
                text=text,
                request_time=response["request_time"],
                error=None,
            )
        except Exception as error:
            raise ValueError(f"Failed to decode tokens with {self.__class__.__name__} tokenizer: {error}") from error

    @abstractmethod
    def _tokenize_do_it(self, request: TokenizationRequest) -> Dict[str, Any]:
        """Callable that tokenizes the text and returns a dictionary with the expected key:
            - "token_ids": a list of tokens ids (expected if `request.encode` is True)
            - "token_strings": a list of tokens strings (expected if `request.encode` is False)
        This function can return both keys if the tokenizer returns both token ids and token strings.
        In that case make sure to set use_encode_in_cache_key to False to avoid double caching.

        Additional keys can be added if a custom `tokenize` method is implemented. Otherwise, the
        default implementation will ignore them.
        """
        pass

    @abstractmethod
    def _decode_do_it(self, request: DecodeRequest) -> Dict[str, Any]:
        """Callable that decodes the tokens and returns a dictionary with the expected key:
            - "text": the decoded text
        Additional keys can be added if a custom `decode` method is implemented. Otherwise, the
        default implementation will ignore them.
        """
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
