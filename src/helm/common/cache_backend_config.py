from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
from typing import Optional
from urllib.parse import SplitResult, urlsplit, urlunsplit

from helm.common.cache import CacheConfig, MongoCacheConfig, BlackHoleCacheConfig, SqliteCacheConfig
from helm.common.general import get_credentials


MONGO_URI_ENV_NAME = "HELM_MONGODB_URI"
MONGO_USERNAME_ENV_NAME = "HELM_MONGODB_USERNAME"
MONGO_PASSWORD_ENV_NAME = "HELM_MONGODB_PASSWORD"

MONGO_URI_CREDENTIALS_KEY = "mongoUri"
MONGO_USERNAME_CREDENTIALS_KEY = "mongoUsername"
MONGO_PASSWORD_CREDENTIALS_KEY = "mongoPassword"


def redact_mongo_uri(uri: str) -> str:
    """Return a MongoDB URI with credentials removed from the authority."""
    parsed: SplitResult = urlsplit(uri)
    if not parsed.username and not parsed.password:
        return uri

    hostname = parsed.hostname or ""
    netloc = f"[{hostname}]" if ":" in hostname else hostname
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))


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

    username: Optional[str] = None
    """Optional MongoDB username passed separately from the URI."""

    password: Optional[str] = None
    """Optional MongoDB password passed separately from the URI."""

    def __repr__(self) -> str:
        username = "<set>" if self.username else None
        password = "<redacted>" if self.password else None
        return (
            f"MongoCacheBackendConfig(uri={redact_mongo_uri(self.uri)!r}, "
            f"username={username!r}, password={password!r})"
        )

    def get_cache_config(self, shard_name: str) -> CacheConfig:
        return MongoCacheConfig(
            uri=self.uri,
            collection_name=shard_name,
            username=self.username,
            password=self.password,
        )


def get_mongo_cache_backend_config(
    mongo_uri: Optional[str],
    base_path: str,
) -> Optional[MongoCacheBackendConfig]:
    """Resolve MongoDB cache settings from CLI args, env vars, and credentials.conf."""
    credentials = get_credentials(base_path)
    resolved_uri = mongo_uri or os.getenv(MONGO_URI_ENV_NAME) or credentials.get(MONGO_URI_CREDENTIALS_KEY, None)
    if not resolved_uri:
        return None

    username = os.getenv(MONGO_USERNAME_ENV_NAME) or credentials.get(MONGO_USERNAME_CREDENTIALS_KEY, None)
    password = os.getenv(MONGO_PASSWORD_ENV_NAME) or credentials.get(MONGO_PASSWORD_CREDENTIALS_KEY, None)
    if bool(username) != bool(password):
        raise ValueError(
            f"Set both {MONGO_USERNAME_ENV_NAME} and {MONGO_PASSWORD_ENV_NAME}, "
            f"or both {MONGO_USERNAME_CREDENTIALS_KEY} and {MONGO_PASSWORD_CREDENTIALS_KEY} "
            "in credentials.conf."
        )

    return MongoCacheBackendConfig(uri=resolved_uri, username=username, password=password)


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
