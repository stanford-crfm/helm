import shutil
import tempfile

from helm.common.cache_backend_config import BlackHoleCacheBackendConfig
from helm.benchmark.window_services.test_utils import get_tokenizer_service, TEST_PROMPT
from helm.benchmark.window_services.tokenizer_service import TokenizerService
from helm.benchmark.window_services.window_service_factory import WindowServiceFactory


class TestYaLMWindowService:
    def setup_method(self):
        self.path: str = tempfile.mkdtemp()
        service: TokenizerService = get_tokenizer_service(self.path, BlackHoleCacheBackendConfig())
        self.window_service = WindowServiceFactory.get_window_service("together/yalm", service)

    def teardown_method(self, method):
        shutil.rmtree(self.path)

    def test_encode(self):
        assert self.window_service.encode(TEST_PROMPT).token_values == [
            496,
            7229,
            388,
            10166,
            397,
            9922,
            64981,
            356,
            4082,
            30020,
            127611,
            360,
            302,
            1330,
            60504,
            25067,
            9006,
            799,
            306,
            272,
            40069,
            10771,
            388,
            15069,
            127602,
            31892,
            2833,
            91312,
            32939,
            356,
            15839,
            127615,
            127611,
            407,
            21317,
            312,
            1460,
            16738,
            27632,
            305,
            272,
            2262,
            127585,
            3706,
            127585,
            321,
            29667,
            306,
            20999,
            7214,
            127581,
        ]

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
            " Artificial",
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

    def test_fits_within_context_window(self):
        # Should fit in the context window since we subtracted the number of tokens of the test prompt
        # from the max request length of 2048
        assert self.window_service.fits_within_context_window(TEST_PROMPT, self.window_service.max_request_length - 51)
        # Should not fit within the max request length because we're expecting one more extra token in the completion
        assert not self.window_service.fits_within_context_window(
            TEST_PROMPT, self.window_service.max_request_length - 51 + 1
        )

    def test_truncate_from_right(self):
        # Create a prompt that exceed max context length: 51 * 41 = 2091 tokens
        long_prompt: str = TEST_PROMPT * 41
        assert not self.window_service.fits_within_context_window(long_prompt)

        # Truncate and ensure it fits within the context window
        truncated_long_prompt: str = self.window_service.truncate_from_right(long_prompt)
        assert self.window_service.get_num_tokens(truncated_long_prompt) == self.window_service.max_request_length
        assert self.window_service.fits_within_context_window(truncated_long_prompt)

    def test_tokenize_and_count(self):
        assert self.window_service.get_num_tokens(TEST_PROMPT) == 51
