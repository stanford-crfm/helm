from tokenize import Token
from typing import List, Optional

from common.tokenization_request import TokenizationRequest, TokenizationRequestResult, TokenizationToken, TextRange
from .tokenizer import Tokenizer
from .tokenizer_service import TokenizerService


class AI21Tokenizer(Tokenizer):
    """Tokenizes by making a request to the proxy server with REST endpoint: `/api/tokenize`."""

    # The max token length of the model input
    MAX_SEQUENCE_LENGTH: int = 2048

    # The max sequence length is the same as the max request length for AI21.
    MAX_REQUEST_LENGTH: int = 2048

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
    def max_request_length(self) -> int:
        # Sometimes splitting a long string to multiple shorter ones introduce new tokens,
        # so we adopt a smaller value here for stability.
        # e.g. "burying him" -> ["_burying"(0,7), "_him"(7,11)];
        # " burying him" -> ["_"(0,0), "_burying"(0,8), "_him"(8,12)];
        # "'s your camera" -> ["▁"(0,0), "'s"(0,2), "▁your▁camera"(2,14)]
        return AI21Tokenizer.MAX_REQUEST_LENGTH - 3

    @property
    def end_of_text_token(self) -> str:
        # TODO: I'm not sure what their end of text token is. I don't think it's documented.
        return " "

    @property
    def prefix_token(self) -> str:
        """AI21 tokenizers do no have a prefix token"""
        return ""

    def encode(self, text: str) -> List:
        """
        Encodes the input text to tokens.
        """
        # If text is empty, skips the API call and returns an empty list.
        if not text:
            return []
        response: TokenizationRequestResult = self._make_tokenization_request(text)
        return response.tokens

    def decode(self, tokens: List, original_text: Optional[str] = None) -> str:
        """
        Given a list of tokens, outputs the corresponding text.
        """
        if not tokens:
            return ""
        # The original text is necessary for decoding AI21 tokens.
        assert original_text
        # The tokens must be a consecutive subset of the original text.
        for i in range(len(tokens) - 1):
            assert tokens[i].text_range.end == tokens[i + 1].text_range.start
        start_token: TokenizationToken = tokens[0]
        end_token: TokenizationToken = tokens[-1]
        start: int = start_token.text_range.start
        end: int = end_token.text_range.end
        return original_text[start:end]

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

    def truncate_from_right(self, text: str, expected_completion_token_length: int = 0) -> str:
        """
        Truncates the text using the AI21 Jurassic tokenizer.
        First tokenizes, then truncates the list of tokens to fit within the context window minus the
        expected completion length (defaults to 0), then uses the start of the text range of the first
        token and the end of the text range of the last token of the truncated list of tokens to
        build the truncated text.
        """
        response: TokenizationRequestResult = self._make_tokenization_request(text)

        # Only look at the first `self.max_sequence_length` - `expected_completion_token_length`
        # number of tokens to the fit the text within the context window.
        # Each token is represented like this: {'text': '▁Hello', 'textRange': {'start': 0, 'end': 5}}
        tokens: List[TokenizationToken] = response.tokens[: self.max_sequence_length - expected_completion_token_length]

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
