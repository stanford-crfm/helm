import tempfile

from helm.benchmark.window_services.test_bloom_window_service import TestBloomWindowService
from helm.benchmark.window_services.window_service_factory import TokenizerService, WindowServiceFactory
from helm.benchmark.window_services.test_utils import get_tokenizer_service


class TestBloomzWindowService(TestBloomWindowService):
    def setup_method(self):
        self.path: str = tempfile.mkdtemp()
        service: TokenizerService = get_tokenizer_service(self.path)
        self.window_service = WindowServiceFactory.get_window_service("together/bloomz", service)
