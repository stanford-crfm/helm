import shutil
import tempfile


from helm.common.cache_backend_config import BlackHoleCacheBackendConfig
from helm.common.local_context import LocalContext
from helm.benchmark.window_services.tokenizer_service import TokenizerService


class TestAdapter:
    """
    Has setup and teardown methods downstream Adapter tests need.
    """

    def setup_method(self):
        self.path: str = tempfile.mkdtemp()
        context = LocalContext(base_path=self.path, cache_backend_config=BlackHoleCacheBackendConfig())
        self.tokenizer_service = TokenizerService(context)

    def teardown_method(self, _):
        shutil.rmtree(self.path)
