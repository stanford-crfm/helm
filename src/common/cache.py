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

        # According to https://github.com/RaRe-Technologies/sqlitedict/issues/145:
        # The code inside the context manager (the with block = one SqliteDict database connection)
        # is thread-safe within a single process.
        #
        # From https://sqlite.org/faq.html#q5:
        # SQLite uses reader/writer locks to control access to the database.
        # SQLite allows multiple processes to have the database file open at once, and for multiple
        # processes to read the database at once. When any process wants to write, it must lock the
        # entire database file for the duration of its update. But that normally only takes a few
        # milliseconds. Other processes just wait on the writer to finish then continue about
        # their business
        #
        # Locking is done for us. From https://www.sqlite.org/lockingv3.html:
        # An EXCLUSIVE lock is needed in order to write to the database file. Only one EXCLUSIVE lock
        # is allowed on the file and no other locks of any kind are allowed to coexist with an EXCLUSIVE
        # lock. In order to maximize concurrency, SQLite works to minimize the amount of time that EXCLUSIVE
        # locks are held.
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
