import os
import tempfile
from typing import List

from helm.common.cache import SqliteCacheConfig
from helm.common.general import parallel_map
from helm.common.tokenization_request import (
    DecodeRequest,
    DecodeRequestResult,
    TokenizationRequest,
    TokenizationRequestResult,
)
from helm.tokenizers.huggingface_tokenizer import HuggingFaceTokenizer


class TestAnthropicTokenizer:
    TEST_PROMPT: str = "I am a computer scientist."
    TEST_ENCODED: List[int] = [45, 1413, 269, 6797, 22228, 18]
    TEST_TOKENS: List[str] = ["I", " am", " a", " computer", " scientist", "."]

    def setup_method(self, method):
        cache_file = tempfile.NamedTemporaryFile(delete=False)
        self.cache_path: str = cache_file.name
        self.tokenizer = HuggingFaceTokenizer(
            SqliteCacheConfig(self.cache_path),
            tokenizer_name="anthropic/claude",
            pretrained_model_name_or_path="Xenova/claude-tokenizer",
        )

    def teardown_method(self, method):
        os.remove(self.cache_path)

    def test_tokenize(self):
        request = TokenizationRequest(text=self.TEST_PROMPT, tokenizer="anthropic/claude")
        result: TokenizationRequestResult = self.tokenizer.tokenize(request)
        assert not result.cached, "First time making the tokenize request. Result should not be cached"
        assert result.raw_tokens == self.TEST_TOKENS
        result = self.tokenizer.tokenize(request)
        assert result.cached, "Result should be cached"
        assert result.raw_tokens == self.TEST_TOKENS

    def test_encode(self):
        request = TokenizationRequest(
            text=self.TEST_PROMPT, tokenizer="anthropic/claude", encode=True, truncation=True, max_length=1
        )
        result: TokenizationRequestResult = self.tokenizer.tokenize(request)
        assert not result.cached, "First time making the tokenize request. Result should not be cached"
        assert result.raw_tokens == [self.TEST_ENCODED[0]]
        result = self.tokenizer.tokenize(request)
        assert result.cached, "Result should be cached"
        assert result.raw_tokens == [self.TEST_ENCODED[0]]

        request = TokenizationRequest(
            text=self.TEST_PROMPT, tokenizer="anthropic/claude", encode=True, truncation=True, max_length=1024
        )
        result = self.tokenizer.tokenize(request)
        assert not result.cached, "First time making this particular request. Result should not be cached"
        assert result.raw_tokens == self.TEST_ENCODED

    def test_decode(self):
        request = DecodeRequest(tokens=self.TEST_ENCODED, tokenizer="anthropic/claude")
        result: DecodeRequestResult = self.tokenizer.decode(request)
        assert not result.cached, "First time making the decode request. Result should not be cached"
        assert result.text == self.TEST_PROMPT
        result = self.tokenizer.decode(request)
        assert result.cached, "Result should be cached"
        assert result.text == self.TEST_PROMPT

    def test_already_borrowed(self):
        """Test workaround of the "Already borrowed" bug (#1421) caused by the thread-hostile Anthropic tokenizer,
        which is a thin wrapper around a Hugging Face FastTokenizer"""

        def make_tokenize_request(seed: int) -> None:
            request_length = 10
            truncation = bool(seed % 2)
            self.tokenizer.tokenize(
                # The truncation parameter requires setting a flag on the Rust FastTokenizer.
                # Concurrent requests cause concurrent mutations, which results an Rust concurrency error.
                TokenizationRequest(
                    text=str(seed) * request_length, tokenizer="anthropic/claude", encode=True, truncation=truncation
                )
            )

        num_requests = 100
        # Should not raise "Already borrowed" error
        parallel_map(make_tokenize_request, list(range(num_requests)), parallelism=8)
