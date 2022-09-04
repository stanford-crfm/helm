import argparse
import json
from collections import defaultdict
from typing import Dict

from sqlitedict import SqliteDict

from common.cache import request_to_key
from common.hierarchical_logger import hlog, htrack


"""
Script to update cache with responses/request results.

Usage:

    python3 scripts/offline_eval/import_results.py <Org - one of 'microsoft' or 'together'> <Path to results jsonl file>

    Examples:

        python3 scripts/offline_eval/import_results.py together results.jsonl
"""


@htrack("Updating cache with requests and results")
def import_results(org: str, request_results_path: str, dry_run: bool):
    """
    Given a jsonl file with request and results, uploads request/result pairs to the cache at `cache_path`.
    We assume each line of the input jsonl file is structured {request: ..., result: ...}.
    """
    count: int = 0
    cache_path: str = f"prod_env/cache/{org}.sqlite"

    # For MT-NLG, we send the same request `num_completions` times because the API does not support the OpenAI
    # parameter 'n'. In our cache, we use `completion_index` to differentiate responses for the same request,
    # We need to keep track of how many times we've seen a request to determine the value of `completion_index`.
    request_counts: Dict[str, int] = defaultdict(int)

    # Updates cache with request/result pairs from input jsonl file at `request_results_path`
    with SqliteDict(cache_path) as cache:
        with open(request_results_path, "r") as f:
            for line in f:
                if len(line.strip()) == 0:
                    continue

                request_and_result: Dict = json.loads(line)
                request: Dict = request_and_result["request"]
                result: Dict = request_and_result["result"]

                if org == "together":
                    # Remove extraneous fields included in the request
                    request.pop("request_type", None)
                    request.pop("model", None)
                    cache[request_to_key(request)] = result
                elif org == "microsoft":
                    # Get the value of `completion_index` which is the current count
                    key: str = request_to_key(request)
                    completion_index: int = request_counts[key]
                    request_counts[key] += 1
                    cache_key: str = request_to_key({"completion_index": completion_index, **request})
                    cache[cache_key] = result

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
    parser.add_argument("org", type=str, help="Org to import results for", choices=["microsoft", "together"])
    parser.add_argument("request_results_path", type=str, help="Path to jsonl file with requests and results.")
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
