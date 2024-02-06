import tempfile

from helm.common.cache_backend_config import BlackHoleCacheBackendConfig
from helm.benchmark.window_services.test_t511b_window_service import TestT511bWindowService
from helm.benchmark.window_services.window_service_factory import TokenizerService, WindowServiceFactory
from helm.benchmark.window_services.test_utils import get_tokenizer_service


class TestFlanT5WindowService(TestT511bWindowService):
    def setup_method(self):
        self.path: str = tempfile.mkdtemp()
        service: TokenizerService = get_tokenizer_service(self.path, BlackHoleCacheBackendConfig())
        self.window_service = WindowServiceFactory.get_window_service("together/flan-t5-xxl", service)
