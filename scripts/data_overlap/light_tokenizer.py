import re

from typing import List
from string import punctuation


class LightTokenizer:
    """
    Tokenize texts by splitting on whitespaces.
    """

    def tokenize(self, text: str) -> List[str]:
        return text.split()


class DefaultTokenizer(LightTokenizer):
    """
    Normalize and tokenize texts by converting all characters to the lower case and
    splitting on whitespaces and punctuations.
    """

    def __init__(self):
        super().__init__()
        self.r = re.compile(r"[\s{}]+".format(re.escape(punctuation)))

    def tokenize(self, text: str) -> List[str]:
        return self.r.split(text.lower())
