import os
import shutil
import tempfile

from helm.common.authentication import Authentication
from helm.proxy.services.service import CACHE_DIR
from helm.proxy.services.server_service import ServerService
from helm.benchmark.window_services.tokenizer_service import TokenizerService


class TestAdapter:
    """
    Has setup and teardown methods downstream Adapter tests need.
    """

    def setup_method(self):
        self.path: str = tempfile.mkdtemp()
        service = ServerService(base_path=self.path, root_mode=True, cache_path=os.path.join(self.path, CACHE_DIR))
        self.tokenizer_service = TokenizerService(service, Authentication("test"))

    def teardown_method(self, _):
        shutil.rmtree(self.path)
