import shutil
import tempfile

from helm.benchmark.window_services.tokenizer_service import TokenizerService
from helm.clients.image_generation.dalle2_client import DALLE2Client
from helm.benchmark.window_services.test_utils import get_tokenizer_service, TEST_PROMPT
from helm.benchmark.window_services.window_service_factory import WindowServiceFactory
from helm.common.cache_backend_config import BlackHoleCacheBackendConfig


class TestOpenAIDALLEWindowService:
    def setup_method(self):
        self.path: str = tempfile.mkdtemp()
        service: TokenizerService = get_tokenizer_service(self.path, BlackHoleCacheBackendConfig())
        self.window_service = WindowServiceFactory.get_window_service("openai/dall-e-2", service)

    def teardown_method(self, method):
        shutil.rmtree(self.path)

    def test_fits_within_context_window(self):
        assert self.window_service.fits_within_context_window(TEST_PROMPT)

    def test_truncate_from_right(self):
        long_prompt: str = TEST_PROMPT * 10
        assert not self.window_service.fits_within_context_window(long_prompt)

        # Truncate and ensure it fits within the context window
        truncated_long_prompt: str = self.window_service.truncate_from_right(long_prompt)
        assert len(truncated_long_prompt) == DALLE2Client.MAX_PROMPT_LENGTH
        assert self.window_service.fits_within_context_window(truncated_long_prompt)
