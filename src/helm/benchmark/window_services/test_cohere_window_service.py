# mypy: check_untyped_defs = False
import os
import shutil
import tempfile
from typing import List

from sqlitedict import SqliteDict

from helm.common.cache_backend_config import SqliteCacheBackendConfig
from helm.common.general import ensure_directory_exists
from .test_cohere_window_service_utils import REQUESTS_TO_RESPONSES, TEST_PROMPT, TOKENIZED_PROMPT
from .tokenizer_service import TokenizerService
from .window_service_factory import WindowServiceFactory
from .test_utils import get_tokenizer_service


class TestCohereWindowService:
    @classmethod
    def setup_class(cls):
        cls.path: str = tempfile.mkdtemp()
        cache_path: str = os.path.join(cls.path, "cache")
        ensure_directory_exists(cache_path)

        # Build the cache with real requests and responses
        with SqliteDict(os.path.join(cache_path, "cohere.sqlite")) as cache:
            for request, response in REQUESTS_TO_RESPONSES.items():
                cache[request] = response
            cache.commit()

        # Requests/responses are already cached. Just write out a fake key to credentials.conf.
        with open(os.path.join(cls.path, "credentials.conf"), "w") as f:
            f.write("cohereApiKey: secret")

        service: TokenizerService = get_tokenizer_service(cls.path, SqliteCacheBackendConfig(cache_path))
        cls.window_service = WindowServiceFactory.get_window_service("cohere/xlarge-20220609", service)
        cls.prompt: str = TEST_PROMPT
        cls.tokenized_prompt: List[str] = TOKENIZED_PROMPT

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.path)

    def test_max_request_length(self):
        assert self.window_service.max_request_length == 2048

    def test_encode(self):
        assert self.window_service.encode(self.prompt).token_values == self.tokenized_prompt

    def test_decode(self):
        assert self.window_service.decode(self.window_service.encode(self.prompt).tokens) == self.prompt

    def test_tokenize(self):
        assert self.window_service.tokenize(self.prompt) == self.tokenized_prompt

    def test_tokenize_and_count(self):
        assert self.window_service.get_num_tokens(self.prompt) == 6

    def test_fits_within_context_window(self):
        # Should fit in the context window since we subtracted the number of tokens of the prompt
        # from the max context window.
        assert self.window_service.fits_within_context_window(self.prompt, self.window_service.max_request_length - 6)
        # Should not fit in the context window because we're expecting one more extra token in the completion.
        assert not self.window_service.fits_within_context_window(
            self.prompt, self.window_service.max_request_length - 6 + 1
        )

    def test_truncate_from_right(self):
        # Create a prompt that exceed max context length: 6 * 342 = 2,052 tokens
        long_prompt: str = self.prompt * 342
        assert not self.window_service.fits_within_context_window(long_prompt)

        # Truncate and ensure it fits within the context window
        truncated_long_prompt: str = self.window_service.truncate_from_right(long_prompt)
        assert self.window_service.get_num_tokens(truncated_long_prompt) == self.window_service.max_request_length
        assert self.window_service.fits_within_context_window(truncated_long_prompt)
