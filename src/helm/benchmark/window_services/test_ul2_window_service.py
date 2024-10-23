import shutil
import tempfile
from typing import List

from helm.common.cache_backend_config import BlackHoleCacheBackendConfig
from helm.benchmark.window_services.tokenizer_service import TokenizerService
from helm.benchmark.window_services.window_service_factory import WindowServiceFactory
from helm.benchmark.window_services.test_utils import get_tokenizer_service, TEST_PROMPT


class TestUL2WindowService:
    TEST_TOKEN_IDS: List[int] = [
        37,
        1166,
        21,
        2200,
        30,
        2941,
        5154,
        7,
        41,
        4545,
        14908,
        61,
        19,
        46,
        3,
        23,
        25503,
        6121,
        2170,
        91,
        13,
        8,
        19796,
        2548,
        21,
        3892,
        18,
        24382,
        15,
        26,
        24714,
        5869,
        2825,
        1433,
        41,
        5478,
        196,
        61,
        24,
        3,
        8345,
        12,
        143,
        4431,
        15895,
        16,
        8,
        810,
        6,
        606,
        6,
        11,
        12001,
        13,
        3361,
        2250,
        5,
    ]

    def setup_method(self):
        self.path: str = tempfile.mkdtemp()
        service: TokenizerService = get_tokenizer_service(self.path, BlackHoleCacheBackendConfig())
        self.window_service = WindowServiceFactory.get_window_service("together/ul2", service)

    def teardown_method(self, method):
        shutil.rmtree(self.path)

    def test_max_request_length(self):
        assert self.window_service.max_request_length == 511

    def test_encode(self):
        assert self.window_service.encode(TEST_PROMPT).token_values == TestUL2WindowService.TEST_TOKEN_IDS

    def test_decode(self):
        assert self.window_service.decode(self.window_service.encode(TEST_PROMPT).tokens) == TEST_PROMPT

    def test_tokenize(self):
        assert self.window_service.tokenize(TEST_PROMPT) == [
            " The",
            " Center",
            " for",
            " Research",
            " on",
            " Foundation",
            " Model",
            "s",
            " (",
            "CR",
            "FM",
            ")",
            " is",
            " an",
            " ",
            "i",
            "nterdisciplinary",
            " initiative",
            " born",
            " out",
            " of",
            " the",
            " Stanford",
            " Institute",
            " for",
            " Human",
            "-",
            "Center",
            "e",
            "d",
            " Artificial",
            " Intel",
            "lig",
            "ence",
            " (",
            "HA",
            "I",
            ")",
            " that",
            " ",
            "aims",
            " to",
            " make",
            " fundamental",
            " advances",
            " in",
            " the",
            " study",
            ",",
            " development",
            ",",
            " and",
            " deployment",
            " of",
            " foundation",
            " models",
            ".",
        ]

    def test_tokenize_and_count(self):
        # There are 57 tokens in `TEST_PROMPT`.
        assert self.window_service.get_num_tokens(TEST_PROMPT) == 57

    def test_fits_within_context_window(self):
        # Should fit in the context window
        assert self.window_service.fits_within_context_window(TEST_PROMPT, self.window_service.max_request_length - 58)

    def test_truncate_from_right(self):
        # Create a prompt that exceed max context length: 57 * 10 = 570 tokens
        long_prompt: str = TEST_PROMPT * 10
        assert not self.window_service.fits_within_context_window(long_prompt)

        # Truncate and ensure it fits within the context window
        truncated_long_prompt: str = self.window_service.truncate_from_right(long_prompt)
        assert self.window_service.get_num_tokens(truncated_long_prompt) == self.window_service.max_request_length
        assert self.window_service.fits_within_context_window(truncated_long_prompt)
