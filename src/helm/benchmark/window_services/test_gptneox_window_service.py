import shutil
import tempfile
from typing import List

from helm.common.cache_backend_config import BlackHoleCacheBackendConfig
from helm.benchmark.window_services.tokenizer_service import TokenizerService
from helm.benchmark.window_services.window_service_factory import WindowServiceFactory
from helm.benchmark.window_services.test_utils import get_tokenizer_service, TEST_PROMPT


class TestGPTNeoXWindowService:
    TEST_TOKEN_IDS: List[int] = [
        510,
        5197,
        323,
        5489,
        327,
        6807,
        31581,
        313,
        1311,
        18278,
        10,
        310,
        271,
        734,
        36078,
        15952,
        5686,
        562,
        273,
        253,
        22681,
        6516,
        323,
        8801,
        14,
        18760,
        2122,
        3975,
        11232,
        19256,
        313,
        10114,
        42,
        10,
        326,
        13698,
        281,
        1056,
        7936,
        16424,
        275,
        253,
        1263,
        13,
        2440,
        13,
        285,
        19007,
        273,
        12153,
        3210,
        15,
    ]

    def setup_method(self):
        self.path: str = tempfile.mkdtemp()
        service: TokenizerService = get_tokenizer_service(self.path, BlackHoleCacheBackendConfig())
        self.window_service = WindowServiceFactory.get_window_service("huggingface/gpt-neox-20b", service)

    def teardown_method(self, method):
        shutil.rmtree(self.path)

    def test_max_request_length(self):
        assert self.window_service.max_request_length == 2049

    def test_encode(self):
        assert self.window_service.encode(TEST_PROMPT).token_values == TestGPTNeoXWindowService.TEST_TOKEN_IDS

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
            " inter",
            "disciplinary",
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
            "Cent",
            "ered",
            " Art",
            "ificial",
            " Intelligence",
            " (",
            "HA",
            "I",
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
        # There are 52 tokens in `TEST_PROMPT`.
        assert self.window_service.get_num_tokens(TEST_PROMPT) == 52

    def test_fits_within_context_window(self):
        # Should fit in the context window since we subtracted the number of tokens of the test prompt
        # from the max context window
        assert self.window_service.fits_within_context_window(TEST_PROMPT, 2049 - 52)
        # Should not fit in the context window because we're expecting one more extra token in the completion
        assert not self.window_service.fits_within_context_window(TEST_PROMPT, 2049 - 52 + 1)

    def test_truncate_from_right(self):
        # Create a prompt that exceed max context length: 52 * 40 = 2080 tokens
        long_prompt: str = TEST_PROMPT * 40
        assert not self.window_service.fits_within_context_window(long_prompt)

        # Truncate and ensure it fits within the context window
        truncated_long_prompt: str = self.window_service.truncate_from_right(long_prompt)
        assert self.window_service.get_num_tokens(truncated_long_prompt) == 2049
        assert self.window_service.fits_within_context_window(truncated_long_prompt)
