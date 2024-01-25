import shutil
import tempfile

from helm.benchmark.window_services.tokenizer_service import TokenizerService

from .test_utils import get_tokenizer_service, TEST_PROMPT, GPT2_TEST_TOKENS, GPT2_TEST_TOKEN_IDS
from .window_service_factory import WindowServiceFactory


class TestGPT2WindowService:
    def setup_method(self):
        self.path: str = tempfile.mkdtemp()
        service: TokenizerService = get_tokenizer_service(self.path)
        self.window_service = WindowServiceFactory.get_window_service("huggingface/gpt2", service)

    def teardown_method(self, method):
        shutil.rmtree(self.path)

    def test_max_request_length(self):
        assert self.window_service.max_request_length == 1025

    def test_encode(self):
        assert self.window_service.encode(TEST_PROMPT).token_values == GPT2_TEST_TOKEN_IDS

    def test_decode(self):
        assert self.window_service.decode(self.window_service.encode(TEST_PROMPT).tokens) == TEST_PROMPT

    def test_tokenize(self):
        assert self.window_service.tokenize(TEST_PROMPT) == GPT2_TEST_TOKENS

    def test_fits_within_context_window(self):
        # Should fit in the context window since we subtracted the number of tokens of the test prompt
        # from the max context window
        assert self.window_service.fits_within_context_window(TEST_PROMPT, 1025 - 51)
        # Should not fit in the context window because we're expecting one more extra token in the completion
        assert not self.window_service.fits_within_context_window(TEST_PROMPT, 1025 - 51 + 1)

    def test_truncate_from_right(self):
        # Create a prompt that exceed max context length: 51 * 41 = 2091 tokens
        long_prompt: str = TEST_PROMPT * 41
        assert not self.window_service.fits_within_context_window(long_prompt)

        # Truncate and ensure it fits within the context window
        truncated_long_prompt: str = self.window_service.truncate_from_right(long_prompt)
        assert self.window_service.get_num_tokens(truncated_long_prompt) == 1025
        assert self.window_service.fits_within_context_window(truncated_long_prompt)

    def test_truncate_from_right_edge_case(self):
        # Example from https://github.com/huggingface/transformers/issues/17682
        problematic_text: str = "their 'studio'"
        assert (
            self.window_service.truncate_from_right(problematic_text, expected_completion_token_length=0)
            == problematic_text
        )

    def test_tokenize_and_count(self):
        assert self.window_service.get_num_tokens(TEST_PROMPT) == 51
