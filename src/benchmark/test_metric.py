from .basic_metrics import get_num_bytes
from common.request import Token


def test_get_num_bytes():
    tokens = [Token(text, 0, {}) for text in ["bytes:\\x99", "Hello", " world", "bytes:\\xe2\\x80"]]
    assert get_num_bytes(tokens) == 14
