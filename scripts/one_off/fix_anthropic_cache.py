import argparse
from typing import Dict

from sqlitedict import SqliteDict

from common.cache import key_to_request
from common.hierarchical_logger import hlog, htrack

"""
Remove entries in the Anthropic cache with `stop=[]`.

Usage:

    python3 scripts/one_off/fix_anthropic_cache.py -p prod_env/cache/anthropic.sqlite

"""


@htrack("Removing entries in the Anthropic cache with `stop=[]`.")
def fix(cache_path: str, dry_run: bool):
    with SqliteDict(cache_path) as cache:
        hlog(f"Found {len(cache)} entries at {cache_path}.")

        count: int = 0
        for key, response in cache.items():
            request: Dict = key_to_request(key)
            if request["stop"] == []:
                del cache[key]
                count += 1

        hlog(f"Deleting {count} entries...")
        if not dry_run:
            # Write to SQLite
            cache.commit()
            hlog(f"Wrote to {cache_path}.")


def main():
    fix(args.cache_path, args.dry_run)
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
