import json
from typing import Dict, Callable, Tuple

from sqlitedict import SqliteDict


def request_to_key(request: Dict) -> str:
    """Normalize a `request` into a `key` so that we can hash using it."""
    return json.dumps(request, sort_keys=True)


def key_to_request(key: str) -> Dict:
    """Convert the normalized version to the request."""
    return json.loads(key)


class Cache(object):
    """
    A cache for request/response pairs.
    The request is a dictionary, so we have to normalize it into a key.
    We use sqlitedict to persist the cache: https://github.com/RaRe-Technologies/sqlitedict.
    """

    def __init__(self, cache_path: str):
        self.cache_path = cache_path

    def get(self, request: Dict, compute: Callable[[], Dict]) -> Tuple[Dict, bool]:
        """Get the result of `request` (by calling `compute` as needed)."""
        key = request_to_key(request)

        with SqliteDict(self.cache_path) as cache:
            response = cache.get(key)
            if response:
                cached = True
            else:
                # Commit the request and response to SQLite
                cache[key] = response = compute()
                cache.commit()
                cached = False
        return response, cached
