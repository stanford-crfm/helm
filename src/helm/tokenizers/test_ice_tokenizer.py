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
from .ice_tokenizer import ICETokenizer


class TestICETokenizer:
    def setup_method(self, method):
        cache_file = tempfile.NamedTemporaryFile(delete=False)
        self.cache_path: str = cache_file.name
        self.tokenizer_name = "TsinghuaKEG/ice"
        self.tokenizer = ICETokenizer(SqliteCacheConfig(self.cache_path))

        # The test cases were created using the examples from https://github.com/THUDM/icetk#tokenization
        self.test_prompt: str = "Hello World! I am icetk."
        self.encoded_test_prompt: List[int] = [39316, 20932, 20035, 20115, 20344, 22881, 35955, 20007]

    def teardown_method(self, method):
        os.remove(self.cache_path)

    def test_tokenize(self):
        request = TokenizationRequest(text=self.test_prompt, tokenizer=self.tokenizer_name)
        result: TokenizationRequestResult = self.tokenizer.tokenize(request)
        assert not result.cached, "First time making the tokenize request. Result should not be cached"
        result: TokenizationRequestResult = self.tokenizer.tokenize(request)
        assert result.cached, "Result should be cached"
        assert result.raw_tokens == [" Hello", " World", "!", " I", " am", " ice", "tk", "."]

    def test_encode(self):
        request = TokenizationRequest(text=self.test_prompt, tokenizer=self.tokenizer_name, encode=True)
        result: TokenizationRequestResult = self.tokenizer.tokenize(request)
        assert result.raw_tokens == self.encoded_test_prompt

    def test_encode_with_truncation(self):
        max_length: int = 3
        request = TokenizationRequest(
            text=self.test_prompt, tokenizer=self.tokenizer_name, encode=True, truncation=True, max_length=max_length
        )
        result: TokenizationRequestResult = self.tokenizer.tokenize(request)
        assert result.raw_tokens == self.encoded_test_prompt[:max_length]

    def test_decode(self):
        request = DecodeRequest(tokens=self.encoded_test_prompt, tokenizer=self.tokenizer_name)
        result: DecodeRequestResult = self.tokenizer.decode(request)
        assert not result.cached, "First time making the decode request. Result should not be cached"
        result: DecodeRequestResult = self.tokenizer.decode(request)
        assert result.cached, "Result should be cached"
        assert result.text == self.test_prompt
