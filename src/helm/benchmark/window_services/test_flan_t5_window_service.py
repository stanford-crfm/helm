import tempfile

from helm.benchmark.config_registry import register_builtin_configs_from_helm_package
from helm.benchmark.window_services.test_t511b_window_service import TestT511bWindowService
from helm.benchmark.window_services.window_service_factory import TokenizerService, WindowServiceFactory
from helm.benchmark.window_services.test_utils import get_tokenizer_service


class TestFlanT5WindowService(TestT511bWindowService):
    def setup_method(self):
        register_builtin_configs_from_helm_package()
        self.path: str = tempfile.mkdtemp()
        service: TokenizerService = get_tokenizer_service(self.path)
        self.window_service = WindowServiceFactory.get_window_service("together/flan-t5-xxl", service)
