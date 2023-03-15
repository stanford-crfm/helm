from typing import List, Optional

from helm.proxy.clients.cohere_client import CohereClient
from .local_window_service import LocalWindowService
from .tokenizer_service import TokenizerService
from .window_service import EncodeResult
from helm.common.tokenization_request import (
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

        response: TokenizationRequestResult
        tokens: List[TokenizationToken] = []
        if truncation or len(text) <= CohereClient.TOKENIZE_API_MAX_TEXT_LENGTH:
            response = self.service.tokenize(
                TokenizationRequest(
                    text,
                    tokenizer=self.tokenizer_name,
                    # The Cohere API does not support decoding, so set `encode` to False to get the value of tokens
                    # as strings so we can simply concatenate them when we need to decode.
                    encode=False,
                    truncation=truncation,
                    max_length=max_length,
                )
            )
            tokens = response.tokens
        else:
            # Perform chunk encoding: Cohere doesn't support long sequences, so break it up into chunks
            # and make a request for each chunk.
            # This can potentially break up valid tokens at the end of the chunk, but the chunk size
            # is large enough that this happens infrequently.
            chunk_size: int = CohereClient.TOKENIZE_API_MAX_TEXT_LENGTH
            for i in range(0, len(text), chunk_size):
                chunk: str = text[i : chunk_size + i]
                response = self.service.tokenize(
                    TokenizationRequest(chunk, tokenizer=self.tokenizer_name, encode=False, truncation=False)
                )
                tokens.extend(response.tokens)

        return EncodeResult(text=text, tokens=tokens)

    def get_num_tokens(self, text: str) -> int:
        """Tokenizes the text and returns the number of tokens."""
        # We need this check since we can't pass in empty string via the `tokenize` endpoint
        if len(text) == 0:
            return 0
        return len(self.encode(text).tokens)

    def decode(self, tokens: List[TokenizationToken], normalized_text: Optional[str] = None) -> str:
        """
        The Cohere API does not support decoding, but we're able to recover the original text from the
        values of the tokens by concatenating them.

        Note this logic currently only works with English text.
        """
        token_strings = []
        for token in tokens:
            assert isinstance(token.value, str)
            token_strings.append(token.value)
        return "".join(token_strings)

    def fits_within_context_window(self, text: str, expected_completion_token_length: int = 0) -> bool:
        """
        Checks if the given text fits within the context window given by `max_request_length`
        taking to account the expected completion length (defaults to 0).

        According to https://docs.cohere.ai/tokenize-reference#request, for tokenize, text: "the string to
        be tokenized, the minimum text length is 1 character, and the maximum text length is 65,536 characters.",
        so first check if the text has fewer than 65,536 characters.
        """
        return (
            len(text) <= CohereClient.TOKENIZE_API_MAX_TEXT_LENGTH
            and self.get_num_tokens(text) + expected_completion_token_length <= self.max_request_length
        )

    def truncate_from_right(self, text: str, expected_completion_token_length: int = 0) -> str:
        """
        Truncates text from the right to fit within the context window given by `max_request_length`
        minus the expected completion length (defaults to 0).
        """
        # First truncate the text so it's within `CohereClient.TOKENIZE_MAX_TEXT_LENGTH` length.
        text = text[: CohereClient.TOKENIZE_API_MAX_TEXT_LENGTH]

        max_length: int = self.max_request_length - expected_completion_token_length
        result: str = self.decode(self.encode(text, truncation=True, max_length=max_length).tokens)

        # HACK: For the vast majority of cases, the above logic works, but it sometimes doesn't work
        # for non-English text, since Cohere technically only supports English at the moment.
        while not self.fits_within_context_window(result, expected_completion_token_length):
            result = result[:-1]

        return result


class CohereCommandWindowService(CohereWindowService):
    def __init__(self, service: TokenizerService):
        super().__init__(service)

    @property
    def max_request_length(self) -> int:
        """
        The max request length of the model. For Cohere, this is the same as the `max_sequence_length`.
        If we exceed the `max_sequence_length`, we get the following error:

        Request failed with too many tokens: total number of tokens (prompt and prediction) cannot
        exceed 2048 - received 2049. Try using a shorter prompt or a smaller max_tokens value.

        For the Command model, in rare situations, the co.tokenize returns a shorter list of tokens
        than the co.generate. This causes sequence length errors for rare inputs. Cohere's advice is
        to reduce the sequence length to 2020 to avoid these issues.
        """
        return 2020
