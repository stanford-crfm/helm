from typing import List
from transformers import GPT2TokenizerFast

from .tokenizer import Tokenizer


class OpenAITokenizer(Tokenizer):

    # The max length of the model input, not the max length of a request (2049).
    MAX_SEQUENCE_LENGTH = 2048

    END_OF_TEXT_TOKEN = "<|endoftext|>"

    def __init__(self):
        # OpenAI uses the same tokenizer for GPT-2 and GPT-3.
        # Weights are cached at ~/.cache/huggingface/transformers.
        self._tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    def encode(self, text: str) -> List[int]:
        """
        Encodes the input text to tokens.
        """
        return self._tokenizer.encode(text)

    def decode(self, tokens: List[int]) -> str:
        """
        Given a list of tokens, outputs the corresponding text.
        """
        return self._tokenizer.decode(tokens, clean_up_tokenization_spaces=False)

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenizes the text.
        """
        return self._tokenizer.tokenize(text)
