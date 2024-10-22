import shutil
import tempfile

from helm.common.cache_backend_config import BlackHoleCacheBackendConfig
from helm.benchmark.window_services.test_utils import get_tokenizer_service, TEST_PROMPT
from helm.benchmark.window_services.tokenizer_service import TokenizerService
from helm.benchmark.window_services.window_service_factory import WindowServiceFactory


class TestOPTWindowService:
    def setup_method(self):
        self.path: str = tempfile.mkdtemp()
        service: TokenizerService = get_tokenizer_service(self.path, BlackHoleCacheBackendConfig())
        self.window_service = WindowServiceFactory.get_window_service("huggingface/opt-175b", service)

    def teardown_method(self, method):
        shutil.rmtree(self.path)

    def test_encode(self):
        assert self.window_service.encode(TEST_PROMPT).token_values == [
            133,
            824,
            13,
            1624,
            15,
            2475,
            32146,
            36,
            9822,
            16523,
            43,
            16,
            41,
            3222,
            32231,
            3893,
            2421,
            66,
            9,
            5,
            8607,
            2534,
            13,
            3861,
            12,
            31230,
            3215,
            27332,
            6558,
            36,
            6826,
            100,
            43,
            14,
            5026,
            7,
            146,
            6451,
            9766,
            11,
            5,
            892,
            6,
            709,
            6,
            8,
            9737,
            9,
            4811,
            3092,
            4,
        ]

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

    def test_tokenize_and_count(self):
        assert self.window_service.get_num_tokens(TEST_PROMPT) == 51

    def test_fits_within_context_window(self):
        # Should fit in the context window since we subtracted the number of tokens of the test prompt
        # from the max request length of 2049
        assert self.window_service.fits_within_context_window(TEST_PROMPT, self.window_service.max_request_length - 51)
        # Should not fit within the max request length because we're expecting one more extra token in the completion
        assert not self.window_service.fits_within_context_window(
            TEST_PROMPT, self.window_service.max_request_length - 51 + 1
        )

    def test_truncate_from_right(self):
        # Create a prompt that exceed max context length
        long_prompt: str = TEST_PROMPT * 45
        assert not self.window_service.fits_within_context_window(long_prompt)

        # Truncate and ensure it fits within the context window
        truncated_long_prompt: str = self.window_service.truncate_from_right(long_prompt)
        assert self.window_service.get_num_tokens(truncated_long_prompt) == self.window_service.max_request_length
        assert self.window_service.fits_within_context_window(truncated_long_prompt)
