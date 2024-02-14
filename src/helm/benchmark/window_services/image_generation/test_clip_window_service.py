import shutil
import tempfile

from helm.benchmark.window_services.tokenizer_service import TokenizerService
from helm.benchmark.window_services.test_utils import get_tokenizer_service
from helm.benchmark.window_services.window_service_factory import WindowServiceFactory
from helm.common.cache_backend_config import BlackHoleCacheBackendConfig


class TestCLIPWindowService:
    def setup_method(self):
        self.path: str = tempfile.mkdtemp()
        service: TokenizerService = get_tokenizer_service(self.path, BlackHoleCacheBackendConfig())
        self.window_service = WindowServiceFactory.get_window_service("huggingface/dreamlike-photoreal-v2-0", service)

    def teardown_method(self, method):
        shutil.rmtree(self.path)

    def test_truncate_from_right(self):
        example_text: str = (
            "an instqrumemnt used for cutting cloth , paper , axdz othr thdin mteroial , "
            "consamistng of two blades lad one on tvopb of the other and fhastned in tle mixdqdjle "
            "so as to bllow them txo be pened and closed by thumb and fitngesr inserted tgrough rings on"
        )
        assert not self.window_service.fits_within_context_window(example_text)

        # Truncate and ensure it fits within the context window
        truncated_prompt: str = self.window_service.truncate_from_right(example_text)
        assert self.window_service.fits_within_context_window(truncated_prompt)
