import pytest

from helm.common.cache import BlackHoleCacheConfig
from helm.common.tokenization_request import (
    DecodeRequest,
    TokenizationRequest,
    TokenizationToken,
)
from helm.tokenizers.cohere_tokenizer import CohereLocalTokenizer


@pytest.mark.models
def test_tokenize():
    tokenizer = CohereLocalTokenizer(api_key=None, cache_config=BlackHoleCacheConfig())
    request = TokenizationRequest(tokenizer="cohere/command", text="otter 🦦")
    result = tokenizer.tokenize(request)
    assert result.success
    assert not result.cached
    assert result.tokens == [TokenizationToken(token) for token in ["ot", "ter", " �", "�", "�"]]


@pytest.mark.models
def test_encode():
    tokenizer = CohereLocalTokenizer(api_key=None, cache_config=BlackHoleCacheConfig())
    request = TokenizationRequest(tokenizer="cohere/command", text="otter 🦦", encode=True)
    result = tokenizer.tokenize(request)
    assert result.success
    assert not result.cached
    assert result.tokens == [TokenizationToken(token) for token in [1741, 1779, 7728, 107, 107]]


@pytest.mark.models
def test_decode():
    tokenizer = CohereLocalTokenizer(api_key=None, cache_config=BlackHoleCacheConfig())
    request = DecodeRequest(tokenizer="cohere/command", tokens=[1741, 1779, 7728, 107, 107])
    result = tokenizer.decode(request)
    assert result.success
    assert not result.cached
    assert result.text == "otter 🦦"
