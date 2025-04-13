import os
import pytest

from helm.common.cache import BlackHoleCacheConfig
from helm.common.tokenization_request import (
    TokenizationRequest,
    TokenizationToken,
)
from helm.tokenizers.grok_tokenizer import GrokAPITokenizer


@pytest.mark.models
def test_tokenize():
    if not os.environ.get("XAI_API_KEY"):
        pytest.skip("No xAI API key found; skipping test")
    tokenizer = GrokAPITokenizer(cache_config=BlackHoleCacheConfig())
    request = TokenizationRequest(tokenizer="xai/grok-3-beta", text="otter 🦦")
    result = tokenizer.tokenize(request)
    assert result.success
    assert not result.cached
    assert result.tokens == [TokenizationToken(token) for token in ["otter", "", "", ""]]


@pytest.mark.models
def test_encode():
    if not os.environ.get("XAI_API_KEY"):
        pytest.skip("No xAI API key found; skipping test")
    tokenizer = GrokAPITokenizer(cache_config=BlackHoleCacheConfig())
    request = TokenizationRequest(tokenizer="xai/grok-3-beta", text="otter 🦦", encode=True)
    result = tokenizer.tokenize(request)
    assert result.success
    assert not result.cached
    assert result.tokens == [TokenizationToken(token) for token in [142507, 11637, 294, 294]]
