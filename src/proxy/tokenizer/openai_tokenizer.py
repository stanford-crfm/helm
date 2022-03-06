from typing import List, Optional
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

    def encode(self, text: str, truncation: bool = False, max_length: Optional[int] = None) -> List[int]:
        """
        Encodes the input text to tokens.
        """
        return self._tokenizer.encode(text, truncation=truncation, max_length=max_length)

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
