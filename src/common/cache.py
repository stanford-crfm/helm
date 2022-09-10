import json
from typing import Dict, Callable, Tuple
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


@retry
def write_to_cache(cache: SqliteDict, key: str, response: Dict) -> bool:
    """
    Write to cache with retry. Returns boolean indicating whether the write was successful or not.
    """
    try:
        cache[key] = response
        cache.commit()
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

    def __init__(self, cache_path: str):
        self.cache_path: str = cache_path

    def get(self, request: Dict, compute: Callable[[], Dict]) -> Tuple[Dict, bool]:
        """Get the result of `request` (by calling `compute` as needed)."""
        cache_stats.increment_query(self.cache_path)
        key = request_to_key(request)

        with SqliteDict(self.cache_path) as cache:
            response = cache.get(key)
            if response:
                cached = True
            else:
                cached = False
                cache_stats.increment_compute(self.cache_path)
                # Compute and commit the request/response to SQLite
                response = compute()
                write_to_cache(cache, key, response)
        return response, cached
