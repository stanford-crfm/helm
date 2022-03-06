from abc import ABC, abstractmethod
from typing import List, Optional


class Tokenizer(ABC):

    # The max length of the model input, not the max length of a request.
    MAX_SEQUENCE_LENGTH: int

    END_OF_TEXT_TOKEN: Optional[str]

    @abstractmethod
    def encode(self, text: str, truncation: bool = False, max_length: Optional[int] = None) -> List[int]:
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
