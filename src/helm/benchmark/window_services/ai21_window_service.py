import re

from typing import List, Optional, Tuple
from urllib.parse import unquote

from helm.common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
    TokenizationToken,
    TextRange,
)
from .window_service import ConfigurableWindowService, EncodeResult, WindowService
from .tokenizer_service import TokenizerService


class AI21WindowService(ConfigurableWindowService):
    """Tokenizes by making a request to the proxy server with REST endpoint: `/api/tokenize`."""

    # AI21's tokenizer API rejects a tokenization request if the input sequence is too long, so
    # we need to set an upper limit for the length of the request. Empirically, if the GPT2 tokenizer tokenizes a
    # sequence to <= 11000 tokens, then it is most likely safe to assume that AI21's tokenization API will
    # process this request.
    MAX_TOKENIZATION_REQUEST_LENGTH: int = 11000

    # The AI21 tokenizer throws the following error when sending a request with text that has too many characters:
    # "Text must be under 100,000 characters (type=value_error)"
    # Sending a request with 100,000 characters seem to work though.
    MAX_CHARACTER_LENGTH: int = 100_000

    NOT_IMPLEMENTED_ERROR_MESSAGE: str = (
        "AI21 only gave API access to their tokenizer, so this method is not supported."
    )

    def __init__(
        self,
        gpt2_window_service: WindowService,
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
        # We need the `TokenizerService` to make requests to the server.
        self.service: TokenizerService = service
        # As explained above, we need a `GPT2WindowService` to help tokenize long text sequences.
        self.gpt2_window_service: WindowService = gpt2_window_service

    def encode(self, text: str, truncation: bool = False, max_length: Optional[int] = None) -> EncodeResult:
        """
        Encodes the input text to tokens.
        """
        tokens: List[TokenizationToken]
        normalized_text: str
        tokens, normalized_text = self._make_long_tokenization_request(text)
        # The end position of the last token should be the end of the text.
        if len(tokens) > 0:
            assert tokens[-1].text_range is not None
            assert tokens[-1].text_range.end == len(normalized_text)

        return EncodeResult(text=normalized_text, tokens=tokens)

    def decode(self, tokens: List[TokenizationToken], normalized_text: Optional[str] = None) -> str:
        """
        Given the model and a list of tokens, outputs the corresponding text.

        For models using the GPT-2 tokenizer, the tokens are integers; for AI21
        models, the tokens are `TokenizationToken`s.

        Some tokenizers (e.g. AI21) normalize the text before encoding it and
        thus require the `normalized_text` for decoding.
        """
        if not tokens:
            return ""

        # `normalized_text` is necessary for decoding AI21 tokens.
        assert normalized_text, "The AI21 tokenizer needs `normalized_text` for decoding"
        for j in range(len(tokens) - 1):
            first_text_range = tokens[j].text_range
            second_text_range = tokens[j + 1].text_range
            assert first_text_range is not None
            assert second_text_range is not None
            assert (
                first_text_range.end == second_text_range.start
            ), "The tokens to be decoded must form a substring of `normalized_text`."

        token_texts: List[str] = []
        # The format of AI21 byte token representations. e.g. <0xE8>
        byte_pattern = "<0x[0-9A-F]{2}>"
        i: int = 0
        while i < len(tokens):
            # If there are byte tokens, aggregates them to a string
            token_value = tokens[i].value
            assert isinstance(token_value, str)
            if re.match(byte_pattern, token_value):
                bytestring = ""
                while i < len(tokens) and re.match(byte_pattern, token_value):
                    # e.g. <0xE8> -> \xE8
                    bytestring += "\\" + token_value[2:-1]
                    i += 1
                # Convert to encoded URI (e.g., %e2%80%99) and decode
                token_text = unquote(bytestring.replace("\\x", "%"))
            # Not a byte token: retrieves the token text based on text_range.
            else:
                token: TokenizationToken = tokens[i]
                assert token.text_range is not None
                token_text = normalized_text[token.text_range.start : token.text_range.end]
                i += 1
            token_texts.append(token_text)
        return "".join(token_texts)

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenizes the text via the /api/tokenize REST endpoint.
        """
        response: TokenizationRequestResult = self._make_tokenization_request(text)
        result = []
        for token in response.tokens:
            assert isinstance(token.value, str)
            result.append(token.value)
        return result

    def get_num_tokens(self, text: str) -> int:
        """Tokenizes the text using the GPT-2 tokenizer and returns the number of tokens."""
        return len(self.tokenize(text))

    def fits_within_context_window(self, text: str, expected_completion_token_length: int = 0) -> bool:
        return (
            len(text) <= AI21WindowService.MAX_CHARACTER_LENGTH
            and self.get_num_tokens(text) + expected_completion_token_length <= self.max_request_length
        )

    def truncate_from_right(self, text: str, expected_completion_token_length: int = 0) -> str:
        """
        Truncates the text using the AI21 Jurassic tokenizer.
        First, ensures the text is shorter than `AI21Tokenizer.MAX_CHARACTER_LENGTH` long.
        Tokenizes, then truncates the list of tokens to fit within the context window minus the
        expected completion length (defaults to 0), then uses the start of the text range of the first
        token and the end of the text range of the last token of the truncated list of tokens to
        build the truncated text.
        """
        text = text[: AI21WindowService.MAX_CHARACTER_LENGTH]
        response: TokenizationRequestResult = self._make_tokenization_request(text)

        # Only look at the first `self.max_request_length` - `expected_completion_token_length`
        # number of tokens to the fit the text within the context window.
        # Each token is represented like this: {'text': '▁Hello', 'textRange': {'start': 0, 'end': 5}}
        max_length: int = self.max_request_length - expected_completion_token_length
        tokens: List[TokenizationToken] = response.tokens[:max_length]

        # If there is no tokens, just return the original text
        if len(tokens) == 0:
            return text

        # AI21 uses "_" to represent a single space in their tokens, so we have to build the new text from the
        # original text after truncation using the text ranges of tokens generated from the original text.
        assert tokens[0].text_range is not None
        first_text_range: TextRange = tokens[0].text_range
        assert tokens[-1].text_range is not None
        last_text_range: TextRange = tokens[-1].text_range
        start: int = first_text_range.start
        end: int = last_text_range.end
        truncated_text: str = text[start:end]

        # HACK: For the vast majority of cases, the above logic works, but there are a few where the
        # token count exceeds `max_length` by 1. This might be a bug with the AI21 tokenizer API.
        # We handle those by removing characters one by one until it fits within the context window.
        while not self.fits_within_context_window(truncated_text, expected_completion_token_length):
            end -= 1
            truncated_text = text[start:end]
        return truncated_text

    def _make_tokenization_request(self, text: str) -> TokenizationRequestResult:
        """Sends a request to the server to tokenize the text via the `TokenizerService`."""
        return self.service.tokenize(TokenizationRequest(text=text, tokenizer=self.tokenizer_name))

    def _make_long_tokenization_request(self, text: str) -> Tuple[List[TokenizationToken], str]:
        """If the text is too long  (longer than 11,000 tokens when tokenized by the GPT-2 tokenizer),
        the AI21 server will close the connection. Therefore, we need to split the text into smaller
        chunks, tokenize each chunk, and re-assemble the tokenization results."""
        # Uses the number of gpt2-style tokens as a measure of text length.
        gpt2_tokens: List[TokenizationToken] = self.gpt2_window_service.encode(text).tokens

        # If the text is short, just makes one request and returns the result.
        if len(gpt2_tokens) < AI21WindowService.MAX_TOKENIZATION_REQUEST_LENGTH:
            result: TokenizationRequestResult = self._make_tokenization_request(text)
            return result.tokens, result.text
        # Otherwise, splits the text to chunks, tokenizes each chunk, and re-assembles them.
        else:
            all_tokens: List[TokenizationToken] = []
            normalized_text_chunks: List[str] = []
            # The number of gpt2-style tokens we have tokenized with the AI21 tokenizer.
            num_processed_tokens: int = 0
            # The length of the (normalized) text string we have tokenized with the AI21 tokenizer.
            num_processed_positions: int = 0
            while num_processed_tokens < len(gpt2_tokens):
                token_chunk_size: int = min(
                    len(gpt2_tokens) - num_processed_tokens, AI21WindowService.MAX_TOKENIZATION_REQUEST_LENGTH
                )
                token_chunk: List[TokenizationToken] = gpt2_tokens[
                    num_processed_tokens : num_processed_tokens + token_chunk_size
                ]
                text_chunk: str = self.gpt2_window_service.decode(token_chunk)
                # We need to avoid generating byte tokens when splitting the text
                while text_chunk.endswith("\ufffd"):
                    token_chunk_size -= 1
                    token_chunk = gpt2_tokens[num_processed_tokens : num_processed_tokens + token_chunk_size]
                    text_chunk = self.gpt2_window_service.decode(token_chunk)
                chunk_result: TokenizationRequestResult = self._make_tokenization_request(text_chunk)
                chunk_tokens: List[TokenizationToken]
                normalized_text_chunk: str
                chunk_tokens, normalized_text_chunk = chunk_result.tokens, chunk_result.text
                # Removes the empty tokens introduced by the split.
                assert chunk_tokens[0].text_range is not None
                if num_processed_tokens != 0 and chunk_tokens[0].text_range.start == chunk_tokens[0].text_range.end:
                    chunk_tokens = chunk_tokens[1:]
                else:
                    chunk_tokens = chunk_tokens[:]

                # Shifts the start and end index of each token
                shifted_tokens: List[TokenizationToken] = []
                for token in chunk_tokens:
                    assert token.text_range is not None
                    shifted_tokens.append(
                        TokenizationToken(
                            value=token.value,
                            text_range=TextRange(
                                start=token.text_range.start + num_processed_positions,
                                end=token.text_range.end + num_processed_positions,
                            ),
                        )
                    )
                all_tokens.extend(shifted_tokens)
                normalized_text_chunks.append(normalized_text_chunk)
                num_processed_tokens += token_chunk_size
                num_processed_positions += len(normalized_text_chunk)

            return all_tokens, "".join(normalized_text_chunks)
