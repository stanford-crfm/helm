from abc import ABC, abstractmethod
from dataclasses import dataclass
import os

from helm.common.cache import CacheConfig, MongoCacheConfig, BlackHoleCacheConfig, SqliteCacheConfig


class CacheBackendConfig(ABC):
    """Config for a cache backend."""

    @abstractmethod
    def get_cache_config(self, shard_name: str) -> CacheConfig:
        """Get a CacheConfig for the given shard."""
        pass


@dataclass(frozen=True)
class MongoCacheBackendConfig(CacheBackendConfig):
    """Config for a MongoDB cache backend."""

    uri: str
    """URL for the MongoDB database that contains the collection.

    Example format: mongodb://[username:password@]host1[:port1]/[dbname]
    For full format, see: https://www.mongodb.com/docs/manual/reference/connection-string/"""

    def get_cache_config(self, shard_name: str) -> CacheConfig:
        return MongoCacheConfig(uri=self.uri, collection_name=shard_name)


@dataclass(frozen=True)
class BlackHoleCacheBackendConfig(CacheBackendConfig):
    """Config for a cache backend that does not save any data."""

    def get_cache_config(self, shard_name: str) -> CacheConfig:
        return BlackHoleCacheConfig()


@dataclass(frozen=True)
class SqliteCacheBackendConfig(CacheBackendConfig):
    """Config for a Sqlite cache backend."""

    path: str
    """Path for the directory that will contain Sqlite files for caches."""

    def get_cache_config(self, shard_name: str) -> CacheConfig:
        return SqliteCacheConfig(path=os.path.join(self.path, f"{shard_name}.sqlite"))
