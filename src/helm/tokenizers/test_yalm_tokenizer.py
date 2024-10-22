# mypy: check_untyped_defs = False
import os
import tempfile
from typing import List

from helm.common.cache import SqliteCacheConfig
from helm.common.tokenization_request import (
    DecodeRequest,
    DecodeRequestResult,
    TokenizationRequest,
    TokenizationRequestResult,
)
from helm.tokenizers.yalm_tokenizer import YaLMTokenizer


class TestYaLMTokenizer:
    def setup_method(self, method):
        cache_file = tempfile.NamedTemporaryFile(delete=False)
        self.cache_path: str = cache_file.name
        self.tokenizer = YaLMTokenizer(SqliteCacheConfig(self.cache_path))

        self.test_prompt: str = "The model leverages 100 billion parameters."
        self.encoded_test_prompt: List[int] = [496, 3326, 30128, 1602, 1830, 8529, 8071, 127581]

    def teardown_method(self, method):
        os.remove(self.cache_path)

    def test_tokenize(self):
        request = TokenizationRequest(tokenizer="Yandex/yalm", text=self.test_prompt)
        result: TokenizationRequestResult = self.tokenizer.tokenize(request)
        assert not result.cached, "First time making the tokenize request. Result should not be cached"
        result: TokenizationRequestResult = self.tokenizer.tokenize(request)
        assert result.cached, "Result should be cached"
        assert result.raw_tokens == [" The", " model", " lever", "ages", " 100", " billion", " parameters", "."]

    def test_encode(self):
        request = TokenizationRequest(tokenizer="Yandex/yalm", text=self.test_prompt, encode=True)
        result: TokenizationRequestResult = self.tokenizer.tokenize(request)
        assert result.raw_tokens == self.encoded_test_prompt

    def test_encode_with_truncation(self):
        max_length: int = 6
        request = TokenizationRequest(
            tokenizer="Yandex/yalm", text=self.test_prompt, encode=True, truncation=True, max_length=max_length
        )
        result: TokenizationRequestResult = self.tokenizer.tokenize(request)
        assert result.raw_tokens == self.encoded_test_prompt[:max_length]

    def test_decode(self):
        request = DecodeRequest(tokenizer="Yandex/yalm", tokens=self.encoded_test_prompt)
        result: DecodeRequestResult = self.tokenizer.decode(request)
        assert not result.cached, "First time making the decode request. Result should not be cached"
        result: DecodeRequestResult = self.tokenizer.decode(request)
        assert result.cached, "Result should be cached"
        assert result.text == self.test_prompt
