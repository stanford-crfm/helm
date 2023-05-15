import shutil
import tempfile

from .test_utils import get_tokenizer_service, TEST_PROMPT, GPT4_TEST_TOKEN_IDS, GPT4_TEST_TOKENS
from .tokenizer_service import TokenizerService
from .window_service_factory import WindowServiceFactory


class TestOpenAIWindowService:
    def setup_method(self):
        self.path: str = tempfile.mkdtemp()
        service: TokenizerService = get_tokenizer_service(self.path)
        self.window_service = WindowServiceFactory.get_window_service("openai/gpt-3.5-turbo-0301", service)

    def teardown_method(self, method):
        shutil.rmtree(self.path)

    def test_encode(self):
        assert self.window_service.encode(TEST_PROMPT).token_values == GPT4_TEST_TOKEN_IDS

    def test_decode(self):
        assert self.window_service.decode(self.window_service.encode(TEST_PROMPT).tokens) == TEST_PROMPT

    def test_tokenize(self):
        assert self.window_service.tokenize(TEST_PROMPT) == GPT4_TEST_TOKENS
