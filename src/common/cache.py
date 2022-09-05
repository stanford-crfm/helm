import json
from typing import Dict, Callable, Tuple

from sqlitedict import SqliteDict

from common.general import hlog
from proxy.retry import get_retry_decorator


def request_to_key(request: Dict) -> str:
    """Normalize a `request` into a `key` so that we can hash using it."""
    return json.dumps(request, sort_keys=True)


def key_to_request(key: str) -> Dict:
    """Convert the normalized version to the request."""
    return json.loads(key)


def retry_if_write_failed(success: bool) -> bool:
    return success


retry: Callable = get_retry_decorator(
    "Write", max_attempts=10, wait_exponential_multiplier_seconds=2, retry_on_result=retry_if_write_failed
)


@retry
def write_to_cache(cache, key, response) -> bool:
    """
    Write to cache with retry. Returns boolean indicating whether the write was successful or not.
    """
    try:
        cache[key] = response
        return True
    except Exception as e:
        hlog(f"Error when writing to cache: {str(e)}")
        return False


class Cache(object):
    """
    A cache for request/response pairs.
    The request is a dictionary, so we have to normalize it into a key.
    We use sqlitedict to persist the cache: https://github.com/RaRe-Technologies/sqlitedict.
    """

    MAX_WRITE_ATTEMPTS: int = 5

    def __init__(self, cache_path: str):
        self.cache_path = cache_path
        # Counters to keep track of progress
        self.num_queries = 0
        self.num_misses = 0

    def get(self, request: Dict, compute: Callable[[], Dict]) -> Tuple[Dict, bool]:
        """Get the result of `request` (by calling `compute` as needed)."""
        self.num_queries += 1
        key = request_to_key(request)

        # TODO: double check autocommit=True
        with SqliteDict(self.cache_path, autocommit=True) as cache:
            response = cache.get(key)
            if response:
                cached = True
            else:
                cached = False
                self.num_misses += 1
                # Compute and commit the request/response to SQLite
                response = compute()
                write_to_cache(cache, key, response)
        return response, cached
