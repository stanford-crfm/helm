import pytest

from helm.common.cache import BlackHoleCacheConfig
from helm.common.tokenization_request import (
    DecodeRequest,
    TokenizationRequest,
    TokenizationToken,
)


@pytest.mark.models
def test_tokenize():
    from helm.tokenizers.ai21_tokenizer import AI21LocalTokenizer

    tokenizer = AI21LocalTokenizer(cache_config=BlackHoleCacheConfig())
    request = TokenizationRequest(tokenizer="ai21/jamba-instruct-tokenizer", text="otter 🦦")
    result = tokenizer.tokenize(request)
    assert result.success
    assert not result.cached
    assert result.tokens == [
        TokenizationToken(token) for token in ["ot", "ter", "▁", "<0xF0>", "<0x9F>", "<0xA6>", "<0xA6>"]
    ]


@pytest.mark.models
def test_encode():
    from helm.tokenizers.ai21_tokenizer import AI21LocalTokenizer

    tokenizer = AI21LocalTokenizer(cache_config=BlackHoleCacheConfig())
    request = TokenizationRequest(tokenizer="ai21/jamba-instruct-tokenizer", text="otter 🦦", encode=True)
    result = tokenizer.tokenize(request)
    assert result.success
    assert not result.cached
    assert result.tokens == [TokenizationToken(token) for token in [1860, 1901, 62934, 1784, 1703, 1710, 1710]]


@pytest.mark.models
def test_decode():
    from helm.tokenizers.ai21_tokenizer import AI21LocalTokenizer

    tokenizer = AI21LocalTokenizer(cache_config=BlackHoleCacheConfig())
    request = DecodeRequest(
        tokenizer="ai21/jamba-instruct-tokenizer", tokens=[1860, 1901, 62934, 1784, 1703, 1710, 1710]
    )
    result = tokenizer.decode(request)
    assert result.success
    assert not result.cached
    assert result.text == "otter 🦦"
