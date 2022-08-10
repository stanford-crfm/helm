import shutil
import tempfile
from typing import List

from .tokenizer_service import TokenizerService
from .window_service_factory import WindowServiceFactory
from .test_utils import get_tokenizer_service, TEST_PROMPT


class TestICEWindowService:
    # According to https://github.com/THUDM/icetk, token id [20100, 83823) are English tokens.
    TEST_TOKEN_IDS: List[int] = [
        20123,
        21490,
        20108,
        22581,
        20111,
        22430,
        48828,
        20019,
        21172,
        27993,
        20014,
        20107,
        20125,
        20105,
        44550,
        27193,
        22258,
        20165,
        20101,
        20100,
        33572,
        22661,
        20108,
        24235,
        20011,
        28882,
        20201,
        59599,
        30558,
        20019,
        68731,
        20014,
        20109,
        24853,
        20103,
        20238,
        24878,
        27849,
        20105,
        20100,
        20299,
        20006,
        20549,
        20006,
        20102,
        28808,
        20101,
        25898,
        21195,
        20007,
    ]

    def setup_method(self):
        self.path: str = tempfile.mkdtemp()
        service: TokenizerService = get_tokenizer_service(self.path)
        self.window_service = WindowServiceFactory.get_window_service("together/glm", service)

    def teardown_method(self, method):
        shutil.rmtree(self.path)

    def test_max_request_length(self):
        assert self.window_service.max_request_length == 2049

    def test_encode(self):
        assert self.window_service.encode(TEST_PROMPT).token_values == TestICEWindowService.TEST_TOKEN_IDS

    def test_decode(self):
        assert self.window_service.decode(self.window_service.encode(TEST_PROMPT).tokens) == TEST_PROMPT

    def test_tokenize(self):
        assert self.window_service.tokenize(TEST_PROMPT) == [
            "▁The",
            "▁Center",
            "▁for",
            "▁Research",
            "▁on",
            "▁Foundation",
            "▁Models",
            "▁(",
            "CR",
            "FM",
            ")",
            "▁is",
            "▁an",
            "▁in",
            "terdisciplinary",
            "▁initiative",
            "▁born",
            "▁out",
            "▁of",
            "▁the",
            "▁Stanford",
            "▁Institute",
            "▁for",
            "▁Human",
            "-",
            "Center",
            "ed",
            "▁Artificial",
            "▁Intelligence",
            "▁(",
            "HAI",
            ")",
            "▁that",
            "▁aims",
            "▁to",
            "▁make",
            "▁fundamental",
            "▁advances",
            "▁in",
            "▁the",
            "▁study",
            ",",
            "▁development",
            ",",
            "▁and",
            "▁deployment",
            "▁of",
            "▁foundation",
            "▁models",
            ".",
        ]

    def test_tokenize_and_count(self):
        # There are 52 tokens in `TEST_PROMPT`.
        assert self.window_service.get_num_tokens(TEST_PROMPT) == 50

    def test_fits_within_context_window(self):
        # Should fit in the context window since we subtracted the number of tokens of the test prompt
        # from the max context window
        assert self.window_service.fits_within_context_window(TEST_PROMPT, self.window_service.max_request_length - 50)
        # Should not fit in the context window because we're expecting one more extra token in the completion
        assert not self.window_service.fits_within_context_window(
            TEST_PROMPT, self.window_service.max_request_length - 50 + 1
        )

    def test_truncate_from_right(self):
        # Create a prompt that exceed max context length: 50 * 42 = 2,100 tokens
        long_prompt: str = TEST_PROMPT * 42
        assert not self.window_service.fits_within_context_window(long_prompt)

        # Truncate and ensure it fits within the context window
        truncated_long_prompt: str = self.window_service.truncate_from_right(long_prompt)
        assert self.window_service.get_num_tokens(truncated_long_prompt) == self.window_service.max_request_length
        assert self.window_service.fits_within_context_window(truncated_long_prompt)
