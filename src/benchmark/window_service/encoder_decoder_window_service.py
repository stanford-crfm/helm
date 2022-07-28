from abc import ABC

from common.tokenization_request import TokenizationRequestResult, TokenizationRequest

from .huggingface_window_service import HuggingFaceWindowService
from .tokenizer_service import TokenizerService
from .window_service import EncodeResult


class EncoderDecoderWindowService(HuggingFaceWindowService, ABC):
    def __init__(self, service: TokenizerService):
        super().__init__(service)

    @property
    def max_request_length(self) -> int:
        """Return the max request length. Account for '</s>'"""
        return self.max_sequence_length + 1

    def fits_within_context_window(self, text: str, expected_completion_token_length: int = 0) -> bool:
        """
        Checks if the given text fits within the context window given by `max_request_length`
        taking to account the expected completion length (defaults to 0).
        """
        return self.get_num_tokens(text) <= self.max_request_length

    def truncate_from_right(self, text: str, expected_completion_token_length: int = 0) -> str:
        """
        Truncates text from the right to fit within the context window given by `max_request_length`
        minus the expected completion length (defaults to 0).

        By default, HuggingFace uses the 'longest_first' truncation strategy:
        "Iteratively reduce the inputs sequence until the input is under max_length starting from the longest one
        at each token (when there is a pair of input sequences)."

        Since we are only passing in a single string, the tokenizer will simply truncate from the right.
        """
        max_length: int = self.max_request_length
        result: str = self.decode(self.encode(text, truncation=True, max_length=max_length).tokens)

        # Validate that the truncated text now fits. Fail fast otherwise.
        num_tokens: int = self.get_num_tokens(result)
        assert num_tokens <= max_length, f"Truncation failed ({num_tokens} > {max_length}). Input text: {text}"
        return result

    def encode(self, text: str, truncation: bool = False, max_length: int = 2048) -> EncodeResult:
        """
        Encodes the input text to tokens.
        """
        response: TokenizationRequestResult = self.service.tokenize(
            TokenizationRequest(
                text, tokenizer=self.tokenizer_name, encode=True, truncation=truncation, max_length=max_length
            )
        )
        # Exclude the '</s>' token that gets added when encoding
        return EncodeResult(text=text, tokens=response.raw_tokens[:-1])
