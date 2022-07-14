import argparse
import json
from typing import Dict

from sqlitedict import SqliteDict

from common.cache import request_to_key
from common.hierarchical_logger import hlog, htrack


"""
Script to update cache with responses/request results for the Together models (together/{model engine}).

Usage:

    python3 scripts/together/together_import_results.py <Path to results jsonl file> <Path to sqlite file for the cache>

    Examples:
        python3 scripts/together/together_import_results.py results.jsonl prod_env/cache/together.sqlite
"""


@htrack("Updating cache with requests and results")
def import_results(cache_path: str, request_results_path: str, dry_run: bool):
    """
    Given a jsonl file with request and results, uploads request/result pairs to the cache at `cache_path`.
    We assume each line of the input jsonl file is structured {request: ..., result: ...}.
    """
    count: int = 0

    # Updates cache with request/result pairs from input jsonl file at `request_results_path`
    with SqliteDict(cache_path) as cache:
        with open(request_results_path, "r") as f:
            for line in f:
                if len(line.strip()) == 0:
                    continue

                request_and_result: Dict = json.loads(line)
                request: Dict = request_and_result["request"]
                result: Dict = request_and_result["result"]

                key: str = request_to_key(request)
                cache[key] = result
                count += 1

                if count > 0 and count % 10_000 == 0:
                    if not dry_run:
                        # Write to SQLite
                        cache.commit()

                    hlog(f"Processed {count} entries")

        if dry_run:
            hlog(f"--dry-run was set. Skipping writing out {count} entries...")
        else:
            # Write to SQLite
            cache.commit()
            hlog(f"Wrote {count} entries to cache at {cache_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("request_results_path", type=str, help="Path to jsonl file with requests and results.")
    parser.add_argument("cache_path", type=str, help="Path to sqlite file for the cache.")
    parser.add_argument(
        "-d",
        "--dry-run",
        action="store_true",
        default=None,
        help="Skips persisting changes to sqlite file and prints log messages instead.",
    )
    args = parser.parse_args()

    import_results(args.cache_path, args.request_results_path, args.dry_run)
    hlog("Done.")
