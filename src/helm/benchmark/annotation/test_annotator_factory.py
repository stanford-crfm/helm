from typing import Any, Dict
import os
import shutil

from helm.benchmark.annotation.annotator_factory import AnnotatorFactory
from helm.benchmark.annotation.annotator import Annotator, AnnotatorSpec
from helm.common.cache_backend_config import BlackHoleCacheBackendConfig


class TestAnnotatorFactory:
    def setup_method(self):
        credentials: Dict[str, Any] = {}
        cache_config = BlackHoleCacheBackendConfig()
        self.file_storage_path: str = "tmp"
        self.annotator_factory = AnnotatorFactory(credentials, self.file_storage_path, cache_config)

    def teardown_method(self):
        if os.path.exists(self.file_storage_path):
            shutil.rmtree(self.file_storage_path)

    def test_get_annotator(self):
        annotator = self.annotator_factory.get_annotator(
            AnnotatorSpec(class_name="helm.benchmark.annotation.annotator.DummyAnnotator")
        )
        assert isinstance(annotator, Annotator)
        assert annotator.name == "dummy"
