from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
from typing import Dict, Callable, Optional, Tuple, Union
from collections import defaultdict
import threading

from sqlitedict import SqliteDict
from common.general import hlog, htrack
from proxy.retry import get_retry_decorator
from bson.son import SON
from pymongo import MongoClient


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
    """Key value store that persists writes."""

    @property
    def path(self):
        return self._path

    def __enter__(self) -> "_KeyValueStore":
        pass

    def __exit__(self, exc_type, exc_value, traceback) -> "_KeyValueStore":
        pass

    @abstractmethod
    def get(self, key: Dict) -> Optional[Dict]:
        pass

    @abstractmethod
    def put(self, key: Dict, value: Dict) -> None:
        pass


class _SqliteKeyValueStore(_KeyValueStore):
    """Key value store backed by a SQLite file."""

    def __init__(self, path: str):
        self._sqlite_dict = SqliteDict(path)
        super().__init__()

    def __enter__(self) -> "_SqliteKeyValueStore":
        self._sqlite_dict.__enter__()
        super().__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> "_SqliteKeyValueStore":
        super().__exit__(exc_type, exc_value, traceback)
        self._sqlite_dict.__exit__(exc_type, exc_value, traceback)
        return self

    def get(self, key: Dict) -> Optional[Dict]:
        key_string = request_to_key(key)
        result = self._sqlite_dict.get(key_string)
        if result is not None:
            assert isinstance(result, dict)
            return result
        return None

    def put(self, key: Dict, value: Dict) -> None:
        key_string = request_to_key(key)
        self._sqlite_dict[key_string] = value
        self._sqlite_dict.commit()


@dataclass(frozen=True)
class MongoConfig:
    """Key value store backed by a MongoDB database."""

    # URL to the MongoDB database.
    # Example format: mongodb://[username:password@]host1[:port1]/[dbname]
    # For full format, see: https://www.mongodb.com/docs/manual/reference/connection-string/
    uri: str

    # Name of the MongoDB collection to use.
    collection_name: str


class _MongoKeyValueStore(_KeyValueStore):
    """Key value store backed by a MongoDB database."""

    _REQUEST_KEY = "request"
    _RESPONSE_KEY = "response"

    def __init__(self, config: MongoConfig):
        # TODO: Create client in __enter__ and clean up client in __exit__
        self._mongodb_client: MongoClient = MongoClient(config.uri)
        self._database = self._mongodb_client.get_default_database()
        self._collection = self._database.get_collection(config.collection_name)
        super().__init__()

    def __enter__(self) -> "_MongoKeyValueStore":
        super().__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> "_MongoKeyValueStore":
        super().__exit__(exc_type, exc_value, traceback)
        return self

    def _canonicalize_key(self, key: Dict) -> SON:
        serialized = json.dumps(key, sort_keys=True)
        return json.loads(serialized, object_pairs_hook=SON)

    def get(self, key: Dict) -> Optional[Dict]:
        query = {self._REQUEST_KEY: self._canonicalize_key(key)}
        request = self._collection.find_one(query)
        if request is not None:
            return request[self._RESPONSE_KEY]
        return None

    def put(self, key: Dict, value: Dict) -> None:
        document = SON([(self._REQUEST_KEY, self._canonicalize_key(key)), (self._RESPONSE_KEY, value)])
        self._collection.insert_one(document)


def _create_key_value_store(path: Union[str, MongoConfig]) -> _KeyValueStore:
    """Create a key value store from the given configuration."""
    # TODO: Support creating _MongoKeyValueStore
    if isinstance(path, MongoConfig):
        return _MongoKeyValueStore(path)
    else:
        return _SqliteKeyValueStore(path)


@retry
def write_to_key_value_store(key_value_store: _KeyValueStore, key: Dict, response: Dict) -> bool:
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

    # Either a string path to the Sqlite file that backs the main cache,
    # or a MongoConfig that specifies a MongoDB database will be used for the main cache
    # instead of SQLite.
    cache_path: Union[str, MongoConfig]

    # Path to the Sqlite file that backs the follower cache.
    # The follower cache is a write-only cache, and responses will not be served from it.
    # Every request and response from the main cache will be written to the follower cache.
    follower_cache_path: Optional[str] = None

    @property
    def cache_stats_key(self):
        if isinstance(self.cache_path, str):
            return self.cache_path
        elif isinstance(self.cache_path, MongoConfig):
            return f"{self.cache_path.uri}/{self.cache_path.collection_name}"


class Cache(object):
    """
    A cache for request/response pairs.
    The request is a dictionary, so we have to normalize it into a key.
    We use sqlitedict to persist the cache: https://github.com/RaRe-Technologies/sqlitedict.
    """

    def __init__(self, config: CacheConfig):
        hlog(f"Created cache with config: {config}")
        self.cache_path: Union[str, MongoConfig] = config.cache_path
        self.follower_cache_path: Optional[str] = config.follower_cache_path
        self.cache_stats_key: str = config.cache_stats_key

    def get(self, request: Dict, compute: Callable[[], Dict]) -> Tuple[Dict, bool]:
        """Get the result of `request` (by calling `compute` as needed)."""
        cache_stats.increment_query(self.cache_stats_key)

        # TODO: Initialize key_value_store in constructor
        with _create_key_value_store(self.cache_path) as key_value_store:
            response = key_value_store.get(request)
            if response:
                cached = True
            else:
                cached = False
                cache_stats.increment_compute(self.cache_stats_key)
                # Compute and commit the request/response to SQLite
                response = compute()

                write_to_key_value_store(key_value_store, request, response)
        if self.follower_cache_path is not None:
            # TODO: Initialize follower_key_value_store in constructor
            with _create_key_value_store(self.follower_cache_path) as follower_key_value_store:
                write_to_key_value_store(follower_key_value_store, request, response)
        return response, cached
