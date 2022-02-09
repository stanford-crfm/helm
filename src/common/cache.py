import json
import threading
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
    Currently, we're just saving everything in a jsonl file.
    """

    def __init__(self, cache_path: str):
        self.cache_path = cache_path
        self._lock = threading.RLock()

    def get(self, request: Dict, compute: Callable[[], Dict]) -> Tuple[Dict, bool]:
        """Get the result of `request` (by calling `compute` as needed)."""
        key = request_to_key(request)

        # According to https://github.com/RaRe-Technologies/sqlitedict/issues/145:
        # The code inside the context manager (the with block = one SqliteDict database connection)
        # is thread-safe within a single process.
        with SqliteDict(self.cache_path) as cache:
            if key in cache:
                response = cache[key]
                cached = True
            else:
                # Commit the request and response to SQLite
                cache[key] = response = compute()
                cache.commit()
                cached = False

        return response, cached
