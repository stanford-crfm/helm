from typing import List, Optional

from proxy.clients.cohere_client import CohereClient
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
        The max length of the model input. Similar to MT-NLG, Cohere does not predict the logprob of
        the first input token so `max_sequence_length` is one token shorter than `max_request_length`.
        """
        return self.max_request_length - 1

    @property
    def max_request_length(self) -> int:
        """
        The max request length of the model. For Cohere, this is the same as the `max_sequence_length`.
        If we exceed the `max_sequence_length`, we get the following error:

        Request failed with too many tokens: total number of tokens (prompt and prediction) cannot
        exceed 2048 - received 2049. Try using a shorter prompt or a smaller max_tokens value.
        """
        return 2048

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
        # Cohere recommended ":", but we can try out different values
        return ":"

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

        Note this logic currently only works with English text.
        """
        return "".join([token.value for token in tokens])

    def fits_within_context_window(self, text: str, expected_completion_token_length: int = 0) -> bool:
        """
        Checks if the given text fits within the context window given by `max_request_length`
        taking to account the expected completion length (defaults to 0).

        According to https://docs.cohere.ai/tokenize-reference#request, for tokenize, text: "the string to
        be tokenized, the minimum text length is 1 character, and the maximum text length is 65536 characters."
        """
        return (
            len(text) <= CohereClient.TOKENIZE_MAX_TEXT_LENGTH
            and self.get_num_tokens(text) + expected_completion_token_length <= self.max_request_length
        )

    def truncate_from_right(self, text: str, expected_completion_token_length: int = 0) -> str:
        """
        Truncates text from the right to fit within the context window given by `max_request_length`
        minus the expected completion length (defaults to 0).
        """
        max_length: int = self.max_request_length - expected_completion_token_length
        result: str = self.decode(self.encode(text, truncation=True, max_length=max_length).tokens)

        # HACK: For the vast majority of cases, the above logic works, but it sometimes doesn't work
        # for non-English text, since Cohere technically only supports English at the moment.
        while not self.fits_within_context_window(result, expected_completion_token_length):
            result = result[:-1]

        return result
