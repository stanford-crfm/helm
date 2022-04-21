import re

from typing import List, Optional, Tuple
from transformers import GPT2TokenizerFast
from urllib.parse import unquote

from common.tokenization_request import TokenizationRequest, TokenizationRequestResult, TokenizationToken, TextRange
from .tokenizer import Tokenizer
from .tokenizer_service import TokenizerService
from .openai_tokenizer import OpenAITokenizer


class AI21Tokenizer(Tokenizer):
    """Tokenizes by making a request to the proxy server with REST endpoint: `/api/tokenize`."""

    # The max token length of the model input
    MAX_SEQUENCE_LENGTH: int = 2048

    # The max sequence length is the same as the max request length for AI21.
    MAX_REQUEST_LENGTH: int = 2048

    # Empirically, if the OpenAI's tokenizer tokenizes a sequence to <= 11000 tokens,
    # then it is most likely safe to assume that AI21's tokenization API will process this request.
    MAX_TOKENIZATION_REQUEST_LENGTH: int = 11000

    NOT_IMPLEMENTED_ERROR_MESSAGE: str = (
        "AI21 only gave API access to their tokenizer, so this method is not supported."
    )

    def __init__(self, model: str, service: TokenizerService):
        self.model: str = model
        # We need the `TokenizerService` to make requests to the server.
        self.service: TokenizerService = service
        self.openai_tokenizer = OpenAITokenizer(GPT2TokenizerFast.from_pretrained("gpt2"))

    @property
    def max_sequence_length(self) -> int:
        return AI21Tokenizer.MAX_SEQUENCE_LENGTH

    @property
    def max_request_length(self) -> int:
        # Sometimes splitting a long string to multiple shorter ones introduce new tokens,
        # so we adopt a smaller value here for stability.
        # e.g. "burying him" -> ["▁burying"(0,7), "▁him"(7,11)];
        # " burying him" -> ["▁"(0,0), "▁burying"(0,8), "▁him"(8,12)];
        # "'s your camera" -> ["▁"(0,0), "'s"(0,2), "▁your▁camera"(2,14)]
        #
        # Also, sometimes a character/word is split into multiple bytes.
        # e.g. '艙' -> ['<0xE8>', '<0x89>', '<0x99>']
        return AI21Tokenizer.MAX_REQUEST_LENGTH - 10

    @property
    def end_of_text_token(self) -> str:
        # TODO: I'm not sure what their end of text token is. I don't think it's documented.
        return " "

    @property
    def prefix_token(self) -> str:
        """AI21 tokenizers do no have a prefix token"""
        return ""

    def encode(self, text: str) -> Tuple[List, str]:
        """
        Encodes the input text to tokens.
        """
        # If text is empty, skips the API call and returns an empty list.
        if not text:
            return [], text
        tokens: List[TokenizationToken]
        normalized_text: str
        tokens, normalized_text = self._make_long_tokenization_request(text)
        # The end position of the last token should be the end of the text.
        assert tokens[-1].text_range.end == len(normalized_text)
        return tokens, normalized_text

    def decode(self, tokens: List, text: Optional[str] = None) -> str:
        """
        Given a list of tokens, outputs the corresponding text.
        """
        if not tokens:
            return ""
        # Text is necessary for decoding AI21 tokens.
        assert text
        # The tokens must be a consecutive subset of the original text.
        for j in range(len(tokens) - 1):
            assert tokens[j].text_range.end == tokens[j + 1].text_range.start, (
                tokens[j].text_range,
                tokens[j + 1].text_range,
            )
        token_texts: List[str] = []
        # The format of AI21 byte token representations. e.g. <0xE8>
        byte_pattern = "<0x[0-9A-F]{2}>"
        i: int = 0
        while i < len(tokens):
            # If there are byte tokens, aggregates them to a string
            if re.match(byte_pattern, tokens[i].text):
                bytestring = ""
                while i < len(tokens) and re.match(byte_pattern, tokens[i].text):
                    # e.g. <0xE8> -> \xE8
                    bytestring += "\\" + tokens[i].text[2:-1]
                    i += 1
                # Convert to encoded URI (e.g., %e2%80%99) and decode
                token_text = unquote(bytestring.replace("\\x", "%"))
            # Not a byte token: retrieves the token text based on text_range.
            else:
                token: TokenizationToken = tokens[i]
                token_text = text[token.text_range.start : token.text_range.end]
                i += 1
            token_texts.append(token_text)
        return "".join(token_texts)

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

    def _make_long_tokenization_request(self, text: str) -> Tuple[List[TokenizationToken], str]:
        """If the text is too long, the AI21 server will close the connection. Therefore,
        we need to split the text into smaller chunks, tokenize each chunk, and re-assemble
        the tokenization results."""
        # Uses the number of OpenAI tokens as a measure of text length.
        open_ai_tokens: List[int]
        open_ai_tokens, _ = self.openai_tokenizer.encode(text)

        # If the text is short, just makes one request and returns the result.
        if len(open_ai_tokens) < AI21Tokenizer.MAX_TOKENIZATION_REQUEST_LENGTH:
            result: TokenizationRequestResult = self._make_tokenization_request(text)
            return result.tokens, result.text
        # Otherwise, splits the text to chunks, tokenizes each chunk, and re-assembles them.
        else:
            all_tokens: List[TokenizationToken] = []
            normalized_text_chunks: List[str] = []
            # The number of OpenAI tokens we have tokenized with the AI21 tokenizer.
            num_processed_tokens: int = 0
            # The length of the (normalized) text string we have tokenized with the AI21 tokenizer.
            num_processed_positions: int = 0
            while num_processed_tokens < len(open_ai_tokens):
                token_chunk_size: int = min(
                    len(open_ai_tokens) - num_processed_tokens, AI21Tokenizer.MAX_TOKENIZATION_REQUEST_LENGTH
                )
                token_chunk: List[int] = open_ai_tokens[num_processed_tokens : num_processed_tokens + token_chunk_size]
                text_chunk: str = self.openai_tokenizer.decode(token_chunk)
                # We need to avoid generating byte tokens when splitting the text
                while text_chunk.endswith("\ufffd"):
                    token_chunk_size -= 1
                    token_chunk = open_ai_tokens[num_processed_tokens : num_processed_tokens + token_chunk_size]
                    text_chunk = self.openai_tokenizer.decode(token_chunk)
                chunk_result: TokenizationRequestResult = self._make_tokenization_request(text_chunk)
                chunk_tokens: List[TokenizationToken]
                normalized_text_chunk: str
                chunk_tokens, normalized_text_chunk = chunk_result.tokens, chunk_result.text
                # Removes the empty tokens introduced by the split.
                if num_processed_tokens != 0 and chunk_tokens[0].text_range.start == chunk_tokens[0].text_range.end:
                    chunk_tokens = chunk_tokens[1:]
                else:
                    chunk_tokens = chunk_tokens[:]

                # Shifts the start and end index of each token
                shifted_tokens: List[TokenizationToken] = [
                    TokenizationToken(
                        text=token.text,
                        text_range=TextRange(
                            start=token.text_range.start + num_processed_positions,
                            end=token.text_range.end + num_processed_positions,
                        ),
                    )
                    for token in chunk_tokens
                ]
                all_tokens.extend(shifted_tokens)
                normalized_text_chunks.append(normalized_text_chunk)
                num_processed_tokens += token_chunk_size
                num_processed_positions += len(normalized_text_chunk)

            # Check whether the tokens are well-formed. Uncomment below for debugging.
            # assert all_tokens[0].text_range.start == 0
            # assert all_tokens[-1].text_range.end == num_processed_positions
            # for i in range(len(all_tokens) - 1):
            #     assert all_tokens[i].text_range.end == all_tokens[i + 1].text_range.start

            return all_tokens, "".join(normalized_text_chunks)
