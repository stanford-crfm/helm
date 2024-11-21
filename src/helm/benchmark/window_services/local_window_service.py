from abc import ABC
from typing import List, Optional, cast

from helm.benchmark.window_services.window_service import ConfigurableWindowService, EncodeResult
from helm.benchmark.window_services.tokenizer_service import TokenizerService
from helm.common.tokenization_request import (
    DecodeRequest,
    DecodeRequestResult,
    TokenizationRequest,
    TokenizationRequestResult,
    TokenizationToken,
)
from helm.clients.client import cleanup_tokens


class LocalWindowService(ConfigurableWindowService, ABC):
    def __init__(
        self,
        service: TokenizerService,
        tokenizer_name: str,
        max_sequence_length: int,
        max_request_length: Optional[int] = None,
        max_sequence_and_generated_tokens_length: Optional[int] = None,
        end_of_text_token: Optional[str] = None,
        prefix_token: Optional[str] = None,
    ):
        super().__init__(
            tokenizer_name=tokenizer_name,
            max_sequence_length=max_sequence_length,
            max_request_length=max_request_length,
            max_sequence_and_generated_tokens_length=max_sequence_and_generated_tokens_length,
            end_of_text_token=end_of_text_token,
            prefix_token=prefix_token,
        )
        self.service: TokenizerService = service

    def encode(self, text: str, truncation: bool = False, max_length: Optional[int] = None) -> EncodeResult:
        """
        Encodes the input text to tokens.
        """
        # If a value for `max_length` is not specified, then set it to the `max_request_length`of the `WindowService`.
        if max_length is None:
            max_length = self.max_request_length

        response: TokenizationRequestResult = self.service.tokenize(
            TokenizationRequest(
                text, tokenizer=self.tokenizer_name, encode=True, truncation=truncation, max_length=max_length
            )
        )
        return EncodeResult(text=text, tokens=response.tokens)

    def decode(self, tokens: List[TokenizationToken], normalized_text: Optional[str] = None) -> str:
        """
        Given the model and a list of tokens, outputs the corresponding text.

        For models using the GPT-2 tokenizer, the tokens are integers; for AI21
        models, the tokens are `TokenizationToken`s.

        Some tokenizers (e.g. AI21) normalize the text before encoding it and
        thus require the `normalized_text` for decoding.
        """
        # For Hugging Face tokenizers, should set `clean_up_tokenization_spaces` to False
        # (https://github.com/huggingface/transformers/issues/17682).
        # If we don't, something like "their 'studio'" becomes "their'studio'" when decoding.
        response: DecodeRequestResult = self.service.decode(
            DecodeRequest(
                [token.value for token in tokens],  # type: ignore
                tokenizer=self.tokenizer_name,
                clean_up_tokenization_spaces=False,
            )
        )
        return response.text

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenizes the text.
        """
        response: TokenizationRequestResult = self.service.tokenize(
            TokenizationRequest(text, tokenizer=self.tokenizer_name)
        )
        tokens: List[str] = cast(List[str], response.raw_tokens)
        tokens = cleanup_tokens(tokens, self.tokenizer_name)
        return tokens

    def get_num_tokens(self, text: str) -> int:
        """Tokenizes the text and returns the number of tokens."""
        return len(self.encode(text).tokens)

    def fits_within_context_window(self, text: str, expected_completion_token_length: int = 0) -> bool:
        """
        Checks if the given text fits within the context window given by `max_request_length`
        taking to account the expected completion length (defaults to 0).
        """
        return self.get_num_tokens(text) + expected_completion_token_length <= self.max_request_length

    def truncate_from_right(self, text: str, expected_completion_token_length: int = 0) -> str:
        """
        Truncates text from the right to fit within the context window given by `max_request_length`
        minus the expected completion length (defaults to 0).

        By default, Hugging Face uses the 'longest_first' truncation strategy:
        "Iteratively reduce the inputs sequence until the input is under max_length starting from the longest one
        at each token (when there is a pair of input sequences)."

        Since we are only passing in a single string, the tokenizer will simply truncate from the right.
        """
        max_length: int = self.max_request_length - expected_completion_token_length
        result: str = self.decode(self.encode(text, truncation=True, max_length=max_length).tokens)

        # HACK: For the vast majority of cases, the above logic works, but it sometimes doesn't work
        # for non-English, non-Chinese text (e.g., Russian from multi_eurlex. See more
        # in https://github.com/stanford-crfm/helm/issues/1448).
        # Truncate by removing character by character until the prompt fits within the context window.
        while not self.fits_within_context_window(result, expected_completion_token_length):
            result = result[:-1]
        return result
