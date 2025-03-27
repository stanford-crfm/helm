import os
from typing import Any, Mapping

from helm.clients.auto_client import AutoClient
from helm.common.credentials_utils import provide_api_key
from helm.common.cache_backend_config import CacheBackendConfig, CacheConfig
from helm.common.hierarchical_logger import hlog
from helm.common.object_spec import create_object, inject_object_spec_args
from helm.benchmark.annotation.annotator import Annotator, AnnotatorSpec


class AnnotatorFactory:
    """Factory for creating annotators."""

    def __init__(
        self, credentials: Mapping[str, Any], file_storage_path: str, cache_backend_config: CacheBackendConfig
    ):
        self.credentials = credentials
        self.file_storage_path = file_storage_path
        self.cache_backend_config = cache_backend_config
        hlog(f"AnnotatorFactory: file_storage_path = {file_storage_path}")
        hlog(f"AnnotatorFactory: cache_backend_config = {cache_backend_config}")

    def get_annotator(self, annotator_spec: AnnotatorSpec) -> Annotator:
        """Return a annotator based on the name."""
        annotator_name: str = annotator_spec.class_name.split(".")[-1].lower().replace("annotator", "")

        cache_config: CacheConfig = self.cache_backend_config.get_cache_config(annotator_name)
        annotator_spec = inject_object_spec_args(
            annotator_spec,
            constant_bindings={
                "cache_config": cache_config,
            },
            provider_bindings={
                "api_key": lambda: provide_api_key(self.credentials, annotator_name),
                "endpoint": lambda: self.credentials.get(annotator_name + "Endpoint", None),
                "file_storage_path": lambda: self._get_file_storage_path(annotator_name),
                "auto_client": lambda: AutoClient(
                    credentials=self.credentials,
                    file_storage_path=self.file_storage_path,
                    cache_backend_config=self.cache_backend_config,
                ),
            },
        )
        return create_object(annotator_spec)

    def _get_file_storage_path(self, annotator_name: str) -> str:
        # Returns the path to use for a local file cache for the given annotator
        local_file_cache_path: str = os.path.join(self.file_storage_path, "output", annotator_name)
        return local_file_cache_path
