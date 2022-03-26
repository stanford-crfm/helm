from typing import List, Optional

from benchmark.tokenizer_service import TokenizerService
from common.tokenization_request import TokenizationRequest, TokenizationRequestResult, TokenizationToken, TextRange
from .tokenizer import Tokenizer


class AI21Tokenizer(Tokenizer):
    """Tokenizes by making a request to the proxy server with REST endpoint: `/api/tokenize`."""

    # The max token length of the model input
    # The max sequence length is the same as the max request length for AI21.
    MAX_SEQUENCE_LENGTH: int = 2048

    NOT_IMPLEMENTED_ERROR_MESSAGE: str = (
        "AI21 only gave API access to their tokenizer, so this method is not supported."
    )

    def __init__(self, model: str, service: TokenizerService):
        self.model: str = model
        # We need the `TokenizerService` to make requests to the server.
        self.service: TokenizerService = service

    @property
    def max_sequence_length(self) -> int:
        return AI21Tokenizer.MAX_SEQUENCE_LENGTH

    @property
    def end_of_text_token(self) -> str:
        # TODO: I'm not sure what their end of text token is. I don't think it's documented.
        return " "

    def encode(self, text: str, truncation: bool = False, max_length: Optional[int] = None) -> List[int]:
        """
        Encodes the input text to tokens.
        Note: AI21 only gave API access to their tokenizer, so this method is not supported.
        """
        raise NotImplementedError(AI21Tokenizer.NOT_IMPLEMENTED_ERROR_MESSAGE)

    def decode(self, tokens: List[int]) -> str:
        """
        Given a list of tokens, outputs the corresponding text.
        Note: AI21 only gave API access to their tokenizer, so this method is not supported.
        """
        raise NotImplementedError(AI21Tokenizer.NOT_IMPLEMENTED_ERROR_MESSAGE)

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenizes the text via the /api/tokenize REST endpoint.
        """
        response: TokenizationRequestResult = self._make_tokenization_request(text)
        return [token.text for token in response.tokens]

    def tokenize_and_count(self, text: str) -> int:
        """Tokenizes the text using the GPT-2 tokenizer and returns the number of tokens."""
        return len(self.tokenize(text))

    def fits_within_context_window(self, text: str, expected_completion_token_length: int = 0) -> bool:
        return self.tokenize_and_count(text) + expected_completion_token_length <= self.max_sequence_length

    def truncate_from_right(self, text: str) -> str:
        """
        Truncates the text using the AI21 Jurassic tokenizer.
        First tokenizes, then truncates the list of tokens to fit within the context window, then uses
        the start of the text range of the first token and the end of the text range of the last token
        of the truncated list of tokens to build the the truncated text.
        """
        response: TokenizationRequestResult = self._make_tokenization_request(text)

        # Only look at the first `self.max_sequence_length` number of tokens
        # to the fit the text within the context window.
        # Each token is represented like this: {'text': '▁Hello', 'textRange': {'start': 0, 'end': 5}}
        tokens: List[TokenizationToken] = response.tokens[: self.max_sequence_length]

        # If there is no tokens, just return the original text
        if len(tokens) == 0:
            return text

        # AI21 uses "_" to represent a single space in their tokens, so we have to build the new text from the
        # original text after truncation using the text ranges of tokens generated from the original text.
        first_text_range: TextRange = tokens[0].text_range
        last_text_range: TextRange = tokens[-1].text_range
        start: int = first_text_range.start
        end: int = last_text_range.end
        return text[start:end]

    def _make_tokenization_request(self, text: str) -> TokenizationRequestResult:
        """Sends a request to the server to tokenize the text via the `TokenizerService`."""
        return self.service.tokenize(TokenizationRequest(text=text, model=self.model))
