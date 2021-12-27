from collections import OrderedDict
import json
import os
from typing import Dict, Tuple, Callable


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

    def read(self):
        """Read cache data from disk."""
        self.data = OrderedDict()
        if os.path.exists(self.cache_path):
            print(f"Reading {self.cache_path}...", end="")
            for line in open(self.cache_path):
                item = json.loads(line)
                self.data[request_to_key(item["request"])] = item["response"]
            print(f"{len(self.data)} entries")

    def write(self):
        """Write cache data to disk (never should need to do this)."""
        with open(self.cache_path, "w") as f:
            print(f"Writing {self.cache_path}...", end="")
            for key, value in self.data.items():
                item = {
                    "request": key_to_request(key),
                    "response": value,
                }
                print(json.dumps(item), file=f)
            print(f"{len(self.data)} entries")

    def get(self, request: Dict, compute: Callable[[], Dict]) -> Dict:
        """Get the result of `request` (by calling `compute` as needed)."""
        key = request_to_key(request)
        if key in self.data:
            response = self.data[key]
            cached = True
        else:
            response = self.data[key] = compute()
            cached = False
            # Just append
            with open(self.cache_path, "a") as f:
                item = {
                    "request": request,
                    "response": response,
                }
                print(json.dumps(item), file=f)
        return response, cached
