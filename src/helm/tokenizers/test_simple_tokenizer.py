from helm.common.tokenization_request import (
    DecodeRequest,
    TokenizationRequest,
    TokenizationToken,
)
from helm.tokenizers.simple_tokenizer import SimpleTokenizer


def test_simple_tokenizer_tokenize():
    tokenizer = SimpleTokenizer()
    request = TokenizationRequest(tokenizer="simple/tokenizer1", text="otter 🦦")
    result = tokenizer.tokenize(request)
    assert result.success
    assert not result.cached
    assert result.tokens == [TokenizationToken(token) for token in ["o", "t", "t", "e", "r", " ", "🦦"]]


def test_simple_tokenizer_encode():
    tokenizer = SimpleTokenizer()
    request = TokenizationRequest(tokenizer="simple/tokenizer1", text="otter 🦦", encode=True)
    result = tokenizer.tokenize(request)
    assert result.success
    assert not result.cached
    assert result.tokens == [TokenizationToken(token) for token in [111, 116, 116, 101, 114, 32, 129446]]


def test_simple_tokenizer_decode():
    tokenizer = SimpleTokenizer()
    request = DecodeRequest(tokenizer="simple/tokenizer1", tokens=[111, 116, 116, 101, 114, 32, 129446])
    result = tokenizer.decode(request)
    assert result.success
    assert not result.cached
    assert result.text == "otter 🦦"
