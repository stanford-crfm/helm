import os
from typing import List, Optional
from transformers import GPT2TokenizerFast

from common.cache import Cache
from .tokenizer import Tokenizer, EncodeResult


class GPT2Tokenizer(Tokenizer):

    # The max request length of GPT2 is MAX_SEQUENCE_LENGTH + 1.
    MAX_REQUEST_LENGTH: int = 1025

    # The max length of the model input. The max sequence length for GPT-2 is 1024.
    MAX_SEQUENCE_LENGTH: int = 1024

    # The end of text token
    END_OF_TEXT_TOKEN: str = "<|endoftext|>"

    GPT2_CACHE_FILE: str = "gpt2_tokenizer.sqlite"

    # TODO: undo none?
    def __init__(self, tokenizer: GPT2TokenizerFast, cache_path: str=None):
        self._tokenizer: GPT2TokenizerFast = tokenizer
        self._cache_path: str = cache_path
        # self.cache = Cache(os.path.join(cache_path, GPT2Tokenizer.GPT2_CACHE_FILE))

    @property
    def max_sequence_length(self) -> int:
        """Return the max sequence length of this tokenizer."""
        return GPT2Tokenizer.MAX_SEQUENCE_LENGTH

    @property
    def max_request_length(self) -> int:
        """Return the max request length of GPT-2."""
        return GPT2Tokenizer.MAX_REQUEST_LENGTH

    @property
    def end_of_text_token(self) -> str:
        """The end of text token."""
        return GPT2Tokenizer.END_OF_TEXT_TOKEN

    @property
    def prefix_token(self) -> str:
        """The prefix token for OPENAI models is the end of text token."""
        return self.end_of_text_token

    def encode(self, text: str) -> EncodeResult:
        """
        Encodes the input text to tokens.
        """
        tokens: List[int] = self._tokenizer.encode(text)
        return EncodeResult(text=text, tokens=tokens)

    def decode(self, tokens: List[int], normalized_text: Optional[str] = None) -> str:
        """
        Given the model and a list of tokens, outputs the corresponding text.

        For models using the GPT-2 tokenizer, the tokens are integers; for AI21
        models, the tokens are `TokenizationToken`s.

        Some tokenizers (e.g. AI21) normalize the text before encoding it and
        thus require the `normalized_text` for decoding.
        """
        return self._tokenizer.decode(tokens, clean_up_tokenization_spaces=False)

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenizes the text.
        """
        return self._tokenizer.tokenize(text)

    def tokenize_and_count(self, text: str) -> int:
        """Tokenizes the text using the GPT-2 tokenizer and returns the number of tokens."""
        def do_it():
            return {"length": len(self.tokenize(text))}

        result, _ = self.cache.get({"operation": "count", "text": text}, do_it)
        return result["length"]

    def fits_within_context_window(self, text: str, expected_completion_token_length: int = 0) -> bool:
        """
        Checks if the given text fits within the context window given by `max_request_length`
        taking to account the expected completion length (defaults to 0).
        """
        return self.tokenize_and_count(text) + expected_completion_token_length <= self.max_request_length

    def truncate_from_right(self, text: str, expected_completion_token_length: int = 0) -> str:
        """
        Truncates text from the right to fit within the context window given by `max_request_length`
        minus the expected completion length (defaults to 0).

        By default, HuggingFace uses the 'longest_first' truncation strategy:
        "Iteratively reduce the inputs sequence until the input is under max_length starting from the longest one
        at each token (when there is a pair of input sequences)."

        Since we are only passing in a single string, the tokenizer will simply truncate from the right.
        """

        def do_it():
            max_length: int = self.max_request_length - expected_completion_token_length
            # Should set clean_up_tokenization_spaces=False: https://github.com/huggingface/transformers/issues/17682
            # If we don't, something like "their 'studio'" becomes "their'studio'" when decoding.
            truncated_text: str = self._tokenizer.decode(
                self._tokenizer.encode(text, truncation=True, max_length=max_length), clean_up_tokenization_spaces=False
            )

            # Validate that the truncated text now fits. Fail fast otherwise.
            num_tokens: int = self.tokenize_and_count(truncated_text)
            assert num_tokens <= max_length, f"Truncation failed ({num_tokens} > {max_length}). Input text: {text}"
            return {"truncated_text": truncated_text}

        cache_key = {
            "operation": "truncate_from_right",
            "text": text,
            "max_request_length": self.max_request_length,
            "expected_completion_token_length": expected_completion_token_length,
        }
        result, _ = self.cache.get(cache_key, do_it)
        return result["truncated_text"]
