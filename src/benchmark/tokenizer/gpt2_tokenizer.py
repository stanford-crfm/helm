from typing import List, Optional

from .tokenizer import Tokenizer, EncodeResult
from .tokenizer_service import TokenizerService
from common.tokenization_request import (
    DecodeRequest,
    DecodeRequestResult,
    TokenizationRequest,
    TokenizationRequestResult,
)


class GPT2Tokenizer(Tokenizer):

    # The max request length of GPT2 is MAX_SEQUENCE_LENGTH + 1.
    MAX_REQUEST_LENGTH: int = 1025

    # The max length of the model input. The max sequence length for GPT-2 is 1024.
    MAX_SEQUENCE_LENGTH: int = 1024

    # The end of text token
    END_OF_TEXT_TOKEN: str = "<|endoftext|>"

    def __init__(self, service: TokenizerService):
        self.service: TokenizerService = service

    @property
    def max_sequence_length(self) -> int:
        """Return the max sequence length of this tokenizer."""
        return GPT2Tokenizer.MAX_SEQUENCE_LENGTH

    @property
    def max_request_length(self) -> int:
        """Return the max request length of GPT-2."""
        return GPT2Tokenizer.MAX_REQUEST_LENGTH

    @property
    def end_of_text_token(self) -> str:
        """The end of text token."""
        return GPT2Tokenizer.END_OF_TEXT_TOKEN

    @property
    def tokenizer_name(self) -> str:
        """Name of the tokenizer to use when sending a request."""
        return "huggingface/gpt2"

    @property
    def prefix_token(self) -> str:
        """The prefix token for OPENAI models is the end of text token."""
        return self.end_of_text_token

    def encode(self, text: str, truncation: bool = False, max_length: int = 2048) -> EncodeResult:
        """
        Encodes the input text to tokens.
        """
        response: TokenizationRequestResult = self.service.tokenize(
            TokenizationRequest(
                text, tokenizer=self.tokenizer_name, encode=True, truncation=truncation, max_length=max_length
            )
        )
        return EncodeResult(text=text, tokens=response.raw_tokens)

    def decode(self, tokens: List[int], normalized_text: Optional[str] = None) -> str:
        """
        Given the model and a list of tokens, outputs the corresponding text.

        For models using the GPT-2 tokenizer, the tokens are integers; for AI21
        models, the tokens are `TokenizationToken`s.

        Some tokenizers (e.g. AI21) normalize the text before encoding it and
        thus require the `normalized_text` for decoding.
        """
        # Should set clean_up_tokenization_spaces=False: https://github.com/huggingface/transformers/issues/17682
        # If we don't, something like "their 'studio'" becomes "their'studio'" when decoding.
        response: DecodeRequestResult = self.service.decode(
            DecodeRequest(tokens, tokenizer=self.tokenizer_name, clean_up_tokenization_spaces=False)
        )
        return response.text

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenizes the text.
        """
        response: TokenizationRequestResult = self.service.tokenize(
            TokenizationRequest(text, tokenizer=self.tokenizer_name)
        )
        return response.raw_tokens

    def tokenize_and_count(self, text: str) -> int:
        """Tokenizes the text using the GPT-2 tokenizer and returns the number of tokens."""
        return len(self.tokenize(text))

    def fits_within_context_window(self, text: str, expected_completion_token_length: int = 0) -> bool:
        """
        Checks if the given text fits within the context window given by `max_request_length`
        taking to account the expected completion length (defaults to 0).
        """
        return self.tokenize_and_count(text) + expected_completion_token_length <= self.max_request_length

    def truncate_from_right(self, text: str, expected_completion_token_length: int = 0) -> str:
        """
        Truncates text from the right to fit within the context window given by `max_request_length`
        minus the expected completion length (defaults to 0).

        By default, HuggingFace uses the 'longest_first' truncation strategy:
        "Iteratively reduce the inputs sequence until the input is under max_length starting from the longest one
        at each token (when there is a pair of input sequences)."

        Since we are only passing in a single string, the tokenizer will simply truncate from the right.
        """
        max_length: int = self.max_request_length - expected_completion_token_length
        result: str = self.decode(self.encode(text, truncation=True, max_length=max_length).tokens)

        # Validate that the truncated text now fits. Fail fast otherwise.
        num_tokens: int = self.tokenize_and_count(result)
        assert num_tokens <= max_length, f"Truncation failed ({num_tokens} > {max_length}). Input text: {text}"
        return result
