from typing import List, Optional

from .local_window_service import LocalWindowService
from .tokenizer_service import TokenizerService
from .window_service import EncodeResult
from common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
    TokenizationToken,
)


class CohereWindowService(LocalWindowService):
    def __init__(self, service: TokenizerService):
        super().__init__(service)

    @property
    def tokenizer_name(self) -> str:
        return "cohere/cohere"

    @property
    def max_sequence_length(self) -> int:
        """
        The max length of the model input.
        """
        return 2048

    @property
    def max_request_length(self) -> int:
        """
        The max request length of the model. For Cohere, this is the same as the `max_sequence_length`.
        If we exceed the `max_sequence_length`, we get the following error:

        Request failed with too many tokens: total number of tokens (prompt and prediction) cannot
        exceed 2048 - received 2049. Try using a shorter prompt or a smaller max_tokens value.
        """
        return self.max_sequence_length

    @property
    def end_of_text_token(self) -> str:
        """
        The end of text token. Cohere does not have one.
        """
        return ""

    @property
    def prefix_token(self) -> str:
        """
        The prefix token. Cohere does not return the log prob for the first token when `echo_prompt` is True.
        """
        # TODO: figure out the best value for this
        return "!"

    def encode(self, text: str, truncation: bool = False, max_length: Optional[int] = None) -> EncodeResult:
        """
        Encodes the input text to tokens.
        """
        if max_length is None:
            max_length = self.max_request_length

        response: TokenizationRequestResult = self.service.tokenize(
            TokenizationRequest(
                # The Cohere API does not support decoding, so set `encode` to False to get the value of tokens
                # as strings so we can simply concatenate them when we need to decode.
                text,
                tokenizer=self.tokenizer_name,
                encode=False,
                truncation=truncation,
                max_length=max_length,
            )
        )
        return EncodeResult(text=text, tokens=response.tokens)

    def decode(self, tokens: List[TokenizationToken], normalized_text: Optional[str] = None) -> str:
        """
        The Cohere API does not support decoding, but we're able to recover the original text from the
        values of the tokens by concatenating them.
        TODO: this does not work for Chinese text. I followed up with them.
        """
        return "".join([token.value for token in tokens])
