import tempfile

from helm.benchmark.window_services.test_t511b_window_service import TestT511bWindowService
from helm.benchmark.window_services.window_service_factory import TokenizerService, WindowServiceFactory
from helm.benchmark.window_services.test_utils import get_tokenizer_service
from helm.benchmark.model_deployment_registry import maybe_register_helm


class TestFlanT5WindowService(TestT511bWindowService):
    def setup_class(self):
        maybe_register_helm()

    def setup_method(self):
        self.path: str = tempfile.mkdtemp()
        service: TokenizerService = get_tokenizer_service(self.path)
        self.window_service = WindowServiceFactory.get_window_service("together/flan-t5-xxl", service)
