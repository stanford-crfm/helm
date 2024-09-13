from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Callable, Generator, Mapping, Tuple
import json
import threading

import sqlite3

from helm.common.general import hlog, htrack
from helm.common.key_value_store import BlackHoleKeyValueStore, KeyValueStore, SqliteKeyValueStore
from helm.proxy.retry import get_retry_decorator

try:
    from cPickle import loads
except ImportError:
    from pickle import loads


def retry_if_write_failed(success: bool) -> bool:
    """Retries when the write fails."""
    return not success


retry: Callable = get_retry_decorator(
    "Write", max_attempts=5, wait_exponential_multiplier_seconds=2, retry_on_result=retry_if_write_failed
)


class CacheConfig:
    """Configuration for a cache."""

    @property
    def cache_stats_key(self) -> str:
        """The string key used by CacheStats to identify this cache."""
        return "unknown"


class KeyValueStoreCacheConfig(CacheConfig):
    """Configuration for a cache backed by a key-value store."""

    # This was originally to distinguish between "primitive" cache configs
    # and "compound" cache configs. But we don't have any "compound" cache configs currently.
    # Hypthetical "compound" example: ReadOnlyCacheConfig(SqliteCacheConfig("path"))
    # TODO: Maybe remove this eventually?
    pass


@dataclass(frozen=True)
class SqliteCacheConfig(KeyValueStoreCacheConfig):
    """Configuration for a cache backed by SQLite."""

    # Path for the Sqlite file that backs the cache.
    path: str

    @property
    def cache_stats_key(self) -> str:
        return self.path


@dataclass(frozen=True)
class BlackHoleCacheConfig(KeyValueStoreCacheConfig):
    """Configuration for a cache that does not save any data."""

    @property
    def cache_stats_key(self) -> str:
        """The string key used by CacheStats to identify this cache."""
        return "disabled_cache"


@dataclass(frozen=True)
class MongoCacheConfig(KeyValueStoreCacheConfig):
    """Configuration for a cache backed by a MongoDB collection."""

    # URL for the MongoDB database that contains the collection.
    # Example format: mongodb://[username:password@]host1[:port1]/[dbname]
    # For full format, see: https://www.mongodb.com/docs/manual/reference/connection-string/
    uri: str

    # Name of the MongoDB collection.
    collection_name: str

    @property
    def cache_stats_key(self) -> str:
        return f"{self.uri}/{self.collection_name}"


def get_all_from_sqlite(path: str) -> Generator[Tuple[Dict, Dict], None, None]:
    """Yields all decoded key, value pairs from the SQLite cache.

    Thread-hostile. Does not load the entire database into memory, unlike SqliteDict.items().
    """
    connection = sqlite3.connect(path)
    cursor = connection.cursor()
    cursor.execute("SELECT key, value FROM unnamed ORDER BY rowid")
    while True:
        row = cursor.fetchone()
        if not row:
            break
        raw_key, raw_value = row
        key: Dict = json.loads(raw_key)
        value: Dict = loads(raw_value)
        yield (key, value)


def create_key_value_store(config: KeyValueStoreCacheConfig) -> KeyValueStore:
    """Create a key value store from the given configuration."""
    # TODO: Support creating _MongoKeyValueStore
    if isinstance(config, MongoCacheConfig):
        from helm.common.mongo_key_value_store import MongoKeyValueStore

        return MongoKeyValueStore(config.uri, config.collection_name)
    elif isinstance(config, SqliteCacheConfig):
        return SqliteKeyValueStore(config.path)
    elif isinstance(config, BlackHoleCacheConfig):
        return BlackHoleKeyValueStore()
    else:
        raise ValueError(f"CacheConfig with unknown type: {config}")


@retry
def write_to_key_value_store(key_value_store: KeyValueStore, key: Mapping, response: Dict) -> bool:
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


class Cache(object):
    """
    A cache for request/response pairs.
    The request is a dictionary, so we have to normalize it into a key.
    We use sqlitedict to persist the cache: https://github.com/RaRe-Technologies/sqlitedict.
    """

    def __init__(self, config: CacheConfig):
        hlog(f"Created cache with config: {config}")
        if isinstance(config, KeyValueStoreCacheConfig):
            self.config = config
        else:
            raise ValueError(f"CacheConfig with unknown type: {config}")

    def get(self, request: Mapping, compute: Callable[[], Dict]) -> Tuple[Dict, bool]:
        """Get the result of `request` (by calling `compute` as needed)."""
        cache_stats.increment_query(self.config.cache_stats_key)

        # TODO: Initialize key_value_store in constructor
        with create_key_value_store(self.config) as key_value_store:
            response = key_value_store.get(request)
            if response:
                cached = True
            else:
                cached = False
                cache_stats.increment_compute(self.config.cache_stats_key)
                # Compute and commit the request/response to SQLite
                response = compute()

                write_to_key_value_store(key_value_store, request, response)
        return response, cached
