from typing import List, Optional

from .window_service import WindowService, EncodeResult
from .tokenizer_service import TokenizerService
from common.tokenization_request import (
    DecodeRequest,
    DecodeRequestResult,
    TokenizationRequest,
    TokenizationRequestResult,
    TokenizationToken,
)


class ICEWindowService(WindowService):
    def __init__(self, service: TokenizerService):
        self.service: TokenizerService = service

    @property
    def tokenizer_name(self) -> str:
        return "TsinghuaKEG/ice"

    @property
    def max_sequence_length(self) -> int:
        """
        The max length of the model input.
        According to https://github.com/THUDM/GLM-130B, the max sequence length is 2048.
        """
        return 2048

    @property
    def max_request_length(self) -> int:
        return self.max_sequence_length

    @property
    def end_of_text_token(self) -> str:
        """The end of text token."""
        return ""

    @property
    def prefix_token(self) -> str:
        """The prefix token"""
        return self.end_of_text_token

    def encode(self, text: str, truncation: bool = False, max_length: Optional[int] = None) -> EncodeResult:
        """
        Encodes the input text to tokens.
        """
        response: TokenizationRequestResult = self.service.tokenize(
            TokenizationRequest(text, tokenizer=self.tokenizer_name, encode=True)
        )
        return EncodeResult(text=text, tokens=response.tokens)

    def decode(self, tokens: List[TokenizationToken], normalized_text: Optional[str] = None) -> str:
        """
        Given the model and a list of tokens, outputs the corresponding text.
        """
        response: DecodeRequestResult = self.service.decode(
            DecodeRequest([token.value for token in tokens], tokenizer=self.tokenizer_name,)
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

    def get_num_tokens(self, text: str) -> int:
        """Tokenizes the text using the Hugging Face tokenizer and returns the number of tokens."""
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
        """
        max_length: int = self.max_request_length - expected_completion_token_length
        result: str = self.decode(self.encode(text).tokens[:max_length])

        # Validate that the truncated text now fits. Fail fast otherwise.
        num_tokens: int = self.get_num_tokens(result)
        assert num_tokens <= max_length, f"Truncation failed ({num_tokens} > {max_length}). Input text: {text}"
        return result
