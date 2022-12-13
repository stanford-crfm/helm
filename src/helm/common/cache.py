from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
from typing import Dict, Callable, Generator, Iterable, Optional, Tuple
from collections import defaultdict
import sqlite3
import threading

from sqlitedict import SqliteDict
from helm.common.general import hlog, htrack
from helm.proxy.retry import get_retry_decorator
from bson.son import SON
from bson.errors import InvalidDocument
from pymongo import MongoClient, ReplaceOne

try:
    from cPickle import loads
except ImportError:
    from pickle import loads


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


class CacheConfig:
    """Configuration for a cache."""

    pass

    @property
    def cache_stats_key(self) -> str:
        """The string key used by CacheStats to identify this cache."""
        return "unknown"


class KeyValueStoreCacheConfig(CacheConfig):
    """Configuration for a cache backed by a key-value store."""

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


@dataclass(frozen=True)
class WithFollowerCacheConfig(CacheConfig):
    """Configuration of a cache backed by a main cache and a follower cache."""

    # Configuration for the main cache.
    # Responses will be written to and served out of this cache.
    main: KeyValueStoreCacheConfig

    # Configuration for the follower cache.
    # The follower cache is a write-only cache. Responses will be written to this cache,
    # but not served out of this cache.
    follower: KeyValueStoreCacheConfig

    @property
    def cache_stats_key(self) -> str:
        return self.main.cache_stats_key


class KeyValueStore(ABC):
    """Key value store that persists writes."""

    @property
    def path(self):
        return self._path

    def __enter__(self) -> "KeyValueStore":
        pass

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        pass

    @abstractmethod
    def contains(self, key: Dict) -> bool:
        pass

    @abstractmethod
    def get(self, key: Dict) -> Optional[Dict]:
        pass

    @abstractmethod
    def get_all(self) -> Generator[Tuple[Dict, Dict], None, None]:
        pass

    @abstractmethod
    def put(self, key: Dict, value: Dict) -> None:
        pass

    @abstractmethod
    def multi_put(self, pairs: Iterable[Tuple[Dict, Dict]]) -> None:
        pass

    @abstractmethod
    def remove(self, key: Dict) -> None:
        pass


class _SqliteKeyValueStore(KeyValueStore):
    """Key value store backed by a SQLite file."""

    def __init__(self, path: str):
        self._sqlite_dict = SqliteDict(path)
        super().__init__()

    def __enter__(self) -> "_SqliteKeyValueStore":
        self._sqlite_dict.__enter__()
        super().__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        super().__exit__(exc_type, exc_value, traceback)
        self._sqlite_dict.__exit__(exc_type, exc_value, traceback)

    def contains(self, key: Dict) -> bool:
        return request_to_key(key) in self._sqlite_dict

    def get(self, key: Dict) -> Optional[Dict]:
        key_string = request_to_key(key)
        result = self._sqlite_dict.get(key_string)
        if result is not None:
            assert isinstance(result, dict)
            return result
        return None

    def get_all(self) -> Generator[Tuple[Dict, Dict], None, None]:
        for key, value in self._sqlite_dict.items():
            yield (key, value)

    def put(self, key: Dict, value: Dict) -> None:
        key_string = request_to_key(key)
        self._sqlite_dict[key_string] = value
        self._sqlite_dict.commit()

    def multi_put(self, pairs: Iterable[Tuple[Dict, Dict]]) -> None:
        for key, value in pairs:
            self.put(key, value)

    def remove(self, key: Dict) -> None:
        del self._sqlite_dict[key]
        self._sqlite_dict.commit()


class _MongoKeyValueStore(KeyValueStore):
    """Key value store backed by a MongoDB database."""

    # The number of documents to return per batch.
    _BATCH_SIZE: int = 8

    _REQUEST_KEY = "request"
    _RESPONSE_KEY = "response"

    def __init__(self, uri: str, collection_name: str):
        # TODO: Create client in __enter__ and clean up client in __exit__
        self._mongodb_client: MongoClient = MongoClient(uri)
        self._database = self._mongodb_client.get_default_database()
        self._collection = self._database.get_collection(collection_name)
        super().__init__()

    def __enter__(self) -> "_MongoKeyValueStore":
        super().__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        super().__exit__(exc_type, exc_value, traceback)

    def _canonicalize_key(self, key: Dict) -> SON:
        serialized = json.dumps(key, sort_keys=True)
        return json.loads(serialized, object_pairs_hook=SON)

    def contains(self, key: Dict) -> bool:
        query = {self._REQUEST_KEY: self._canonicalize_key(key)}
        return self._collection.find_one(query) is not None

    def get(self, key: Dict) -> Optional[Dict]:
        query = {self._REQUEST_KEY: self._canonicalize_key(key)}
        document = self._collection.find_one(query)
        if document is not None:
            response = document[self._RESPONSE_KEY]
            if isinstance(response, str):
                return json.loads(response)
            else:
                return response
        return None

    def get_all(self) -> Generator[Tuple[Dict, Dict], None, None]:
        for document in self._collection.find({}).batch_size(self._BATCH_SIZE):
            request = document[self._REQUEST_KEY]
            response = document[self._RESPONSE_KEY]
            if isinstance(response, str):
                yield (request, json.loads(response))
            else:
                yield (request, response)

    def put(self, key: Dict, value: Dict) -> None:
        request = self._canonicalize_key(key)
        document = SON([(self._REQUEST_KEY, request), (self._RESPONSE_KEY, value)])
        # The MongoDB collection should have a unique indexed on "request"
        try:
            self._collection.replace_one(filter={"request": request}, replacement=document, upsert=True)
        except InvalidDocument:
            # If the document is malformed e.g. because of null bytes in keys, instead store the response as a string.
            alternate_document = SON([(self._REQUEST_KEY, request), (self._RESPONSE_KEY, json.dumps(value))])
            self._collection.replace_one(filter={"request": request}, replacement=alternate_document, upsert=True)

    def multi_put(self, pairs: Iterable[Tuple[Dict, Dict]]) -> None:
        operations = []
        for key, value in pairs:
            request = self._canonicalize_key(key)
            document = SON([(self._REQUEST_KEY, request), (self._RESPONSE_KEY, value)])
            operations.append(ReplaceOne({self._REQUEST_KEY: request}, document, upsert=True))
        # Note: unlike put, multi_put does not support documents with null bytes in keys.
        self._collection.bulk_write(operations)

    def remove(self, key: Dict) -> None:
        self._collection.delete_one(key)


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
        return _MongoKeyValueStore(config.uri, config.collection_name)
    elif isinstance(config, SqliteCacheConfig):
        return _SqliteKeyValueStore(config.path)
    else:
        raise ValueError(f"KeyValueStoreCacheConfig with unknown type: {config}")


@retry
def write_to_key_value_store(key_value_store: KeyValueStore, key: Dict, response: Dict) -> bool:
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
        self.config: KeyValueStoreCacheConfig
        self.follower_config: Optional[KeyValueStoreCacheConfig]
        if isinstance(config, KeyValueStoreCacheConfig):
            self.config = config
            self.follower_config = None
        elif isinstance(config, WithFollowerCacheConfig):
            self.config = config.main
            self.follower_config = config.follower
        else:
            raise ValueError(f"CacheConfig with unknown type: {config}")

    def get(self, request: Dict, compute: Callable[[], Dict]) -> Tuple[Dict, bool]:
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
        if self.follower_config is not None:
            # TODO: Initialize follower_key_value_store in constructor
            with create_key_value_store(self.follower_config) as follower_key_value_store:
                write_to_key_value_store(follower_key_value_store, request, response)
        return response, cached
