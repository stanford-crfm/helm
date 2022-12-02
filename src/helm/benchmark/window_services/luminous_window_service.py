from abc import abstractmethod
from typing import List, Optional

from .local_window_service import LocalWindowService
from .tokenizer_service import TokenizerService
from .window_service import EncodeResult
from helm.common.tokenization_request import TokenizationRequest, TokenizationToken, DecodeRequest


class LuminousWindowService(LocalWindowService):
    def __init__(self, service: TokenizerService):
        super().__init__(service)

    @property
    @abstractmethod
    def tokenizer_name(self) -> str:
        """Each Luminous model has its own tokenizer."""
        pass

    @property
    def max_sequence_length(self) -> int:
        return self.max_request_length

    @property
    def max_request_length(self) -> int:
        """
        The max request length according to https://docs.aleph-alpha.com/api/complete.
        TODO: double check if this is only for the completion (not including the prompt).
        """
        return 2048

    @property
    def end_of_text_token(self) -> str:
        """
        The end of text token.
        TODO: echo doesn't seem to be supported.
        """
        return ""

    @property
    def prefix_token(self) -> str:
        """
        The prefix token.
        TODO: echo doesn't seem to be supported.
        """
        return self.end_of_text_token

    def encode(self, text: str, truncation: bool = False, max_length: Optional[int] = None) -> EncodeResult:
        """
        Encodes the input text to tokens.
        """
        if max_length is None:
            max_length = self.max_request_length

        response = self.service.tokenize(
            TokenizationRequest(
                text,
                tokenizer=self.tokenizer_name,
                encode=True,
                truncation=truncation,
                max_length=max_length,
            )
        )
        return EncodeResult(text=text, tokens=response.tokens)

    def get_num_tokens(self, text: str) -> int:
        """Tokenizes the text and returns the number of tokens."""
        return len(self.encode(text).tokens)

    def decode(self, tokens: List[TokenizationToken], normalized_text: Optional[str] = None) -> str:
        """
        Decodes the token ids to text.
        """
        response = self.service.decode(
            DecodeRequest(
                tokens=[token.value for token in tokens],  # type: ignore
                tokenizer=self.tokenizer_name,
            )
        )
        return response.text

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
        return self.decode(self.encode(text, truncation=True, max_length=max_length).tokens)


class LuminousBaseWindowService(LuminousWindowService):
    @property
    def tokenizer_name(self) -> str:
        return "AlephAlpha/luminous-base"


class LuminousExtendedWindowService(LuminousWindowService):
    @property
    def tokenizer_name(self) -> str:
        return "AlephAlpha/luminous-extended"


class LuminousSupremeWindowService(LuminousWindowService):
    @property
    def tokenizer_name(self) -> str:
        return "AlephAlpha/luminous-supreme"


class LuminousWorldWindowService(LuminousWindowService):
    @property
    def tokenizer_name(self) -> str:
        return "AlephAlpha/luminous-world"
