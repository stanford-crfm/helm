import argparse
import os

from common.cache import Cache

"""
Script to fix an existing AI21 cache.

Usage:

    python3 scripts/fix_ai21_cache.py -p prod_env/cache/ai21.sqlite

"""

QUOTA_EXCEEDED_ERROR = "Quota exceeded."


def remove_quota_exceeded_entries(cache_path: str):
    cache = Cache(cache_path)

    # Create a copy of keys because we can't delete keys while iterating over an ordered dict
    for key in list(cache.data):
        response = cache.data[key]

        # Remove entries from the cache that has the 'Quota exceeded' error
        if "detail" in response and response["detail"] == QUOTA_EXCEEDED_ERROR:
            del cache.data[key]

    # Delete the old cache file and write out the repaired cache to cache_path
    os.remove(cache_path)
    cache.write()


def main():
    remove_quota_exceeded_entries(args.cache_path)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--cache-path", type=str, default="prod_env/cache/ai21.sqlite", help="Path to AI21 cache.",
    )
    args = parser.parse_args()

    main()
