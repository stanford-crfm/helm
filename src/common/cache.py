from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
from typing import Any, Dict, Callable, Optional, Tuple
from urllib.parse import urlparse
from collections import defaultdict
import threading

from sqlitedict import SqliteDict

from common.general import hlog, htrack
from proxy.retry import get_retry_decorator


def request_to_key(request: Dict) -> str:
    """Normalize a `request` into a `key` so that we can hash using it."""
    return json.dumps(request, sort_keys=True)


def key_to_request(key: str) -> Dict:
    """Convert the normalized version to the request."""
    return json.loads(key)


def retry_if_write_failed(success: bool) -> bool:
    """Retries when the write fails."""
    return not success


retry: Callable = get_retry_decorator(
    "Write", max_attempts=10, wait_exponential_multiplier_seconds=2, retry_on_result=retry_if_write_failed
)


class _KeyValueStore(ABC):
    def __init__(self, path: str):
        self._path = path

    @property
    def path(self):
        return self._path

    def __enter__(self) -> "_KeyValueStore":
        pass

    def __exit__(self, *exc_details) -> "_KeyValueStore":
        pass

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        pass

    @abstractmethod
    def put(self, key: str, value: Any) -> None:
        pass


class _SqliteKeyValueStore(_KeyValueStore):
    def __init__(self, path: str):
        self._sqlite_dict = SqliteDict(path)
        super().__init__(path)

    def __enter__(self) -> "_SqliteKeyValueStore":
        self._sqlite_dict.__enter__()
        return self

    def __exit__(self, *exc_details) -> "_SqliteKeyValueStore":
        self._sqlite_dict.__exit__(*exc_details)
        return self

    def get(self, key: str) -> Optional[Any]:
        result = self._sqlite_dict.get(key)
        return result

    def put(self, key: str, value: Any) -> None:
        self._sqlite_dict[key] = value
        self._sqlite_dict.commit()


def _create_key_value_store(path: str) -> _KeyValueStore:
    parse_result = urlparse(path)
    if parse_result.scheme == "" or parse_result.scheme == "file":
        return _SqliteKeyValueStore(parse_result.path)
    else:
        raise ValueError(f"Cache path contained unsupported scheme: {parse_result.scheme}")


@retry
def write_to_key_value_store(key_value_store: _KeyValueStore, key: str, response: Dict) -> bool:
    """
    Write to the key value store with retry. Returns boolean indicating whether the write was successful or not.
    """
    try:
        key_value_store.put(key, response)
        return True
    except Exception as e:
        hlog(f"Error when writing to cache: {str(e)}")
        return False


class CacheStats:
    """Keeps track of the number of queries and cache misses for various caches."""

    def __init__(self):
        # For each path, how many times did I query the cache?
        self.num_queries: Dict[str, int] = defaultdict(int)

        # For each path, how many times did I miss the cache and have to compute?
        self.num_computes: Dict[str, int] = defaultdict(int)

        self.lock = threading.Lock()

    def reset(self):
        with self.lock:
            self.num_queries.clear()
            self.num_computes.clear()

    def increment_query(self, path: str):
        with self.lock:
            self.num_queries[path] += 1

    def increment_compute(self, path: str):
        with self.lock:
            self.num_computes[path] += 1

    @htrack(None)
    def print_status(self):
        with self.lock:
            for path in self.num_queries:
                hlog(f"{path}: {self.num_queries[path]} queries, {self.num_computes[path]} computes")


# Singleton that's updated from everywhere.  Caches are often deep inside
# various abstractions in the code and inaccessible, so we aggregate statistics
# at the top-level to be able to print all of them out in one place.
cache_stats = CacheStats()


@dataclass(frozen=True)
class CacheConfig:
    """Configuration for a cache."""

    # Path to the Sqlite file that backs the main cache.
    cache_path: str

    # Path to the Sqlite file that backs the follower cache.
    # The follower cache is a write-only cache, and responses will not be served from it.
    # Every request and response from the main cache will be written to the follower cache.
    follower_cache_path: Optional[str] = None


class Cache(object):
    """
    A cache for request/response pairs.
    The request is a dictionary, so we have to normalize it into a key.
    We use sqlitedict to persist the cache: https://github.com/RaRe-Technologies/sqlitedict.
    """

    def __init__(self, config: CacheConfig):
        self._key_value_store: _KeyValueStore = _create_key_value_store(config.cache_path)
        self._follower_key_value_store: Optional[_KeyValueStore] = None
        if config.follower_cache_path:
            self._follower_key_value_store = _create_key_value_store(config.follower_cache_path)

    def get(self, request: Dict, compute: Callable[[], Dict]) -> Tuple[Dict, bool]:
        """Get the result of `request` (by calling `compute` as needed)."""
        cache_stats.increment_query(self._key_value_store.path)
        key = request_to_key(request)

        with self._key_value_store as key_value_store:
            response = key_value_store.get(key)
            if response:
                cached = True
            else:
                cached = False
                cache_stats.increment_compute(key_value_store.path)
                # Compute and commit the request/response to SQLite
                response = compute()
                write_to_key_value_store(key_value_store, key, response)
        if self._follower_key_value_store is not None:
            with self._follower_key_value_store as follower_key_value_store:
                write_to_key_value_store(follower_key_value_store, key, response)
        return response, cached
