import argparse
from typing import Dict

from sqlitedict import SqliteDict

from common.cache import key_to_request, request_to_key
from common.hierarchical_logger import hlog, htrack

"""
Script to add completion_index to the keys of SQLiteDict.

Usage:

    python3 scripts/add_completion_index_to_cache.py -p prod_env/cache/anthropic.sqlite
    python3 scripts/add_completion_index_to_cache.py -p prod_env/cache/microsoft.sqlite

"""


@htrack("Add default completion_index to every key")
def add_completion_index(cache_path: str, dry_run: bool):
    cache_copy = dict()

    with SqliteDict(cache_path) as cache:
        hlog(f"Found {len(cache)} entries at {cache_path}.")

        for key, response in cache.items():
            # Construct the new key with completion_index=0
            request: Dict = key_to_request(key)
            request["completion_index"] = 0
            new_key: str = request_to_key(request)
            cache_copy[new_key] = response

            # Remove the old entry
            del cache[key]

        for key, response in cache_copy.items():
            cache[key] = response
        hlog(f"Modified {len(cache)} entries.")

        if not dry_run:
            # Write to SQLite
            cache.commit()
            hlog(f"Wrote to {cache_path}.")


def main():
    add_completion_index(args.cache_path, args.dry_run)
    hlog("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--cache-path", type=str, help="Path to cache.")
    parser.add_argument(
        "-d",
        "--dry-run",
        action="store_true",
        default=None,
        help="Skips persisting fixes and just outputs log messages.",
    )
    args = parser.parse_args()

    main()
