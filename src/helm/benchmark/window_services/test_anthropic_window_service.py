import shutil
import tempfile
from typing import List

from helm.common.cache_backend_config import BlackHoleCacheBackendConfig
from helm.benchmark.window_services.tokenizer_service import TokenizerService
from helm.benchmark.window_services.window_service_factory import WindowServiceFactory
from helm.benchmark.window_services.test_utils import get_tokenizer_service, TEST_PROMPT


class TestAnthropicWindowService:
    TEST_PROMPT_LENGTH: int = 52
    TEST_TOKEN_IDS: List[int] = [
        773,
        7793,
        351,
        9134,
        440,
        5384,
        34812,
        344,
        2334,
        16598,
        13,
        365,
        471,
        969,
        53329,
        25203,
        6948,
        648,
        316,
        279,
        30839,
        9290,
        351,
        12467,
        17,
        14126,
        3242,
        5995,
        21216,
        32163,
        344,
        7790,
        45,
        13,
        427,
        24442,
        317,
        1400,
        14943,
        29527,
        300,
        279,
        2929,
        16,
        4424,
        16,
        329,
        26398,
        316,
        18200,
        1220,
        18,
    ]
    TEST_TOKENS: List[str] = [
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

    def setup_method(self):
        self.path: str = tempfile.mkdtemp()
        service: TokenizerService = get_tokenizer_service(self.path, BlackHoleCacheBackendConfig())
        self.window_service = WindowServiceFactory.get_window_service("anthropic/claude-v1.3", service)

    def teardown_method(self, method):
        shutil.rmtree(self.path)

    def test_max_request_length(self):
        assert self.window_service.max_request_length == 8000

    def test_encode(self):
        assert self.window_service.encode(TEST_PROMPT).token_values == self.TEST_TOKEN_IDS

    def test_decode(self):
        assert self.window_service.decode(self.window_service.encode(TEST_PROMPT).tokens) == TEST_PROMPT

    def test_tokenize(self):
        assert self.window_service.tokenize(TEST_PROMPT) == self.TEST_TOKENS

    def test_tokenize_and_count(self):
        assert self.window_service.get_num_tokens(TEST_PROMPT) == self.TEST_PROMPT_LENGTH

    def test_fits_within_context_window(self):
        # Should fit in the context window since we subtracted the number of tokens of the test prompt
        # from the max context window
        assert self.window_service.fits_within_context_window(
            TEST_PROMPT, self.window_service.max_request_length - self.TEST_PROMPT_LENGTH
        )
        # Should not fit in the context window because we're expecting one more extra token in the completion
        assert not self.window_service.fits_within_context_window(
            TEST_PROMPT, self.window_service.max_request_length - self.TEST_PROMPT_LENGTH + 1
        )

    def test_truncate_from_right(self):
        # Create a prompt that exceed max context length
        long_prompt: str = TEST_PROMPT * 200
        assert not self.window_service.fits_within_context_window(long_prompt)

        # Truncate and ensure it fits within the context window
        truncated_long_prompt: str = self.window_service.truncate_from_right(long_prompt)
        assert self.window_service.get_num_tokens(truncated_long_prompt) == self.window_service.max_request_length
        assert self.window_service.fits_within_context_window(truncated_long_prompt)
