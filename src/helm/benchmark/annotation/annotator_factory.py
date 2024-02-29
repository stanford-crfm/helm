import os
from typing import Any, Dict, Mapping, Optional

from helm.common.file_caches.file_cache import FileCache
from helm.common.file_caches.local_file_cache import LocalFileCache
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
        self.annotators: Dict[str, Annotator] = {}
        hlog(f"AnnotatorFactory: file_storage_path = {file_storage_path}")
        hlog(f"AnnotatorFactory: cache_backend_config = {cache_backend_config}")

    def get_annotator(self, annotator_spec: AnnotatorSpec) -> Annotator:
        """Return a annotator based on the name."""
        # First try to find the annotator in the cache
        annotator_name: str = annotator_spec.name
        annotator: Optional[Annotator] = self.annotators.get(annotator_name)
        if annotator is not None:
            return annotator

        # Otherwise, create the client
        cache_config: CacheConfig = self.cache_backend_config.get_cache_config(annotator_name)
        annotator_spec = inject_object_spec_args(
            annotator_spec,
            constant_bindings={
                "cache_config": cache_config,
            },
            provider_bindings={
                "api_key": lambda: provide_api_key(self.credentials, annotator_name),
                "file_cache_path": lambda: self._get_file_cache_path(annotator_name),
            },
        )
        annotator = create_object(annotator_spec)

        # Cache the client
        self.annotators[annotator_name] = annotator

        return annotator

    def _get_file_cache(self, annotator_name: str) -> FileCache:
        # Initialize `FileCache` for text-to-image model APIs
        local_file_cache_path: str = os.path.join(self.file_storage_path, "output", annotator_name)
        return LocalFileCache(local_file_cache_path, file_extension="png")

    def _get_file_cache_path(self, annotator_name: str) -> str:
        # Initialize `FileCache` for text-to-image model APIs
        local_file_cache_path: str = os.path.join(self.file_storage_path, "output", annotator_name)
        return local_file_cache_path
