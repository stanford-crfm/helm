import json
import os
import threading
from collections import OrderedDict
from typing import Dict, Callable, Tuple

from sqlitedict import SqliteDict

from common.hierarchical_logger import hlog


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
        self.read()
        self._lock = threading.RLock()

    def read(self):
        """Read cache data from disk."""
        self.data = OrderedDict()
        if os.path.exists(self.cache_path):
            hlog(f"Reading {self.cache_path}...")
            with SqliteDict(self.cache_path) as cache_dict:
                for key, response in cache_dict.items():
                    self.data[key] = response
            hlog(f"{len(self.data)} entries")

    def write(self):
        """Write cache data to disk (never should need to do this)."""
        with SqliteDict(self.cache_path) as cache_dict:
            hlog(f"Writing {self.cache_path}...")
            for key, response in self.data.items():
                cache_dict[key] = response
            cache_dict.commit()
            hlog(f"{len(self.data)} entries")

    def get(self, request: Dict, compute: Callable[[], Dict]) -> Tuple[Dict, bool]:
        """Get the result of `request` (by calling `compute` as needed)."""
        key = request_to_key(request)
        if key in self.data:
            response = self.data[key]
            cached = True
        else:
            response = self.data[key] = compute()
            cached = False

            # Cache new request and response.
            # We acquire the lock before executing the following operation just in case
            # it is not thread-safe.
            # TODO: Check if the following operation is thread-safe. If it is, remove the lock.
            #       https://github.com/RaRe-Technologies/sqlitedict/issues/145
            with self._lock:
                with SqliteDict(self.cache_path) as cache_dict:
                    cache_dict[key] = response
                    cache_dict.commit()
        return response, cached
