from helm.common.request import Token
from helm.benchmark.metrics.basic_metrics import get_num_bytes, convert_tokens_to_text


def test_get_num_bytes():
    tokens = [Token(text, 0) for text in ["bytes:\\x99", "Hello", " world", "bytes:\\xe2\\x80"]]
    assert get_num_bytes(tokens) == 14


def test_convert_tokens_to_text():
    tokens = [
        Token(text, 0)
        for text in [
            "<|endoftext|>",
            "bytes:\\xe2\\x80",
            "bytes:\\x99",
            "Hello",
            " world",
            "bytes:\\xe2\\x80",
            "bytes:\\x99",
            "<|endoftext|>",
        ]
    ]

    groups = convert_tokens_to_text(tokens)
    assert "".join([group["text"] for group in groups]) == "<|endoftext|>’Hello world’<|endoftext|>"
