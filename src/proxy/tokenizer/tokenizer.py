from abc import ABC, abstractmethod
from typing import List


class Tokenizer(ABC):

    # The max length of the model input, not the max length of a request.
    MAX_SEQUENCE_LENGTH: int

    END_OF_TEXT_TOKEN: str

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """
        Encodes the input text to tokens.
        """
        pass

    @abstractmethod
    def decode(self, tokens: List[int]) -> str:
        """
        Given a list of tokens, outputs the corresponding text.
        """
        pass

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenizes the text.
        """
        pass
