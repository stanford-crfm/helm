import shutil
import tempfile
from typing import List

from helm.common.cache_backend_config import BlackHoleCacheBackendConfig
from helm.benchmark.window_services.tokenizer_service import TokenizerService
from helm.benchmark.window_services.window_service_factory import WindowServiceFactory
from helm.benchmark.window_services.test_utils import get_tokenizer_service, TEST_PROMPT


class TestBloomWindowService:
    TEST_TOKEN_IDS: List[int] = [
        2175,
        27149,
        613,
        30469,
        664,
        16289,
        168358,
        375,
        12990,
        76143,
        12,
        632,
        660,
        168734,
        1912,
        51298,
        34181,
        1800,
        461,
        368,
        112640,
        31036,
        613,
        22256,
        7833,
        21830,
        376,
        200008,
        116891,
        375,
        43,
        19540,
        12,
        861,
        83174,
        427,
        5219,
        20079,
        136458,
        361,
        368,
        12589,
        15,
        11468,
        15,
        530,
        94369,
        461,
        88734,
        20038,
        17,
    ]

    def setup_method(self):
        self.path: str = tempfile.mkdtemp()
        service: TokenizerService = get_tokenizer_service(self.path, BlackHoleCacheBackendConfig())
        self.window_service = WindowServiceFactory.get_window_service("together/bloom", service)

    def teardown_method(self, method):
        shutil.rmtree(self.path)

    def test_max_request_length(self):
        assert self.window_service.max_request_length == 2049

    def test_encode(self):
        assert self.window_service.encode(TEST_PROMPT).token_values == TestBloomWindowService.TEST_TOKEN_IDS

    def test_decode(self):
        assert self.window_service.decode(self.window_service.encode(TEST_PROMPT).tokens) == TEST_PROMPT

    def test_tokenize(self):
        assert self.window_service.tokenize(TEST_PROMPT) == [
            "The",
            " Center",
            " for",
            " Research",
            " on",
            " Foundation",
            " Models",
            " (",
            "CR",
            "FM",
            ")",
            " is",
            " an",
            " interdisciplin",
            "ary",
            " initiative",
            " born",
            " out",
            " of",
            " the",
            " Stanford",
            " Institute",
            " for",
            " Human",
            "-C",
            "enter",
            "ed",
            " Artificial",
            " Intelligence",
            " (",
            "H",
            "AI",
            ")",
            " that",
            " aims",
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
        # There are 51 tokens in `TEST_PROMPT`.
        assert self.window_service.get_num_tokens(TEST_PROMPT) == 51

    def test_fits_within_context_window(self):
        # Should fit in the context window since we subtracted the number of tokens of the test prompt
        # from the max context window
        assert self.window_service.fits_within_context_window(TEST_PROMPT, self.window_service.max_request_length - 51)
        # Should not fit in the context window because we're expecting one more extra token in the completion
        assert not self.window_service.fits_within_context_window(
            TEST_PROMPT, self.window_service.max_request_length - 51 + 1
        )

    def test_truncate_from_right(self):
        # Create a prompt that exceed max context length
        long_prompt: str = TEST_PROMPT * 41
        assert not self.window_service.fits_within_context_window(long_prompt)

        # Truncate and ensure it fits within the context window
        truncated_long_prompt: str = self.window_service.truncate_from_right(long_prompt)
        assert self.window_service.get_num_tokens(truncated_long_prompt) == self.window_service.max_request_length
        assert self.window_service.fits_within_context_window(truncated_long_prompt)
