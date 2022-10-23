import argparse
import json
from collections import defaultdict
import os
from typing import Dict, List, Tuple

from common.cache import (
    KeyValueStoreCacheConfig,
    MongoCacheConfig,
    SqliteCacheConfig,
    create_key_value_store,
    request_to_key,
)
from common.hierarchical_logger import hlog, htrack


"""
Script to update cache with responses/request results.

Usage:

    python3 scripts/offline_eval/import_results.py <Org - one of 'microsoft' or 'together'> <Path to results jsonl file>

    Examples:

        python3 scripts/offline_eval/import_results.py together results.jsonl
"""


@htrack("Updating cache with requests and results")
def import_results(cache_config: KeyValueStoreCacheConfig, organization: str, request_results_path: str, dry_run: bool):
    """
    Given a jsonl file with request and results, uploads request/result pairs to the cache at `cache_path`.
    We assume each line of the input jsonl file is structured {request: ..., result: ...}.
    """
    count: int = 0

    # For MT-NLG, we send the same request `num_completions` times because the API does not support the OpenAI
    # parameter 'n'. In our cache, we use `completion_index` to differentiate responses for the same request,
    # We need to keep track of how many times we've seen a request to determine the value of `completion_index`.
    request_counts: Dict[str, int] = defaultdict(int)

    # Updates cache with request/result pairs from input jsonl file at `request_results_path`
    with create_key_value_store(cache_config) as store:
        with open(request_results_path, "r") as f:
            items_to_write: List[Tuple[Dict, Dict]] = []
            for line in f:
                if len(line.strip()) == 0:
                    continue

                request_and_result: Dict = json.loads(line)
                request: Dict = request_and_result["request"]
                result: Dict = request_and_result["result"]

                if organization == "together":
                    # Remove extraneous fields ("request_type" and "model") included in the request
                    # and set "engine" to the value of "model".
                    request.pop("request_type", None)
                    request["engine"] = request.pop("model")
                    items_to_write.append((request, result))
                elif organization == "microsoft":
                    # Get the value of `completion_index` which is the current count
                    key: str = request_to_key(request)
                    completion_index: int = request_counts[key]
                    request_counts[key] += 1
                    cache_key: dict = {"completion_index": completion_index, **request}
                    items_to_write.append((cache_key, result))

                count += 1
                if count > 0 and count % 10_000 == 0:
                    if not dry_run and items_to_write:
                        # Write to SQLite
                        store.multi_put(items_to_write)
                        items_to_write = []

                    hlog(f"Processed {count} entries")

        if dry_run:
            hlog(f"--dry-run was set. Skipping writing out {count} entries...")
        else:
            # Write to SQLite
            if items_to_write:
                store.multi_put(items_to_write)
            hlog(f"Wrote {count} entries to cache at {cache_config.cache_stats_key}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-dir", type=str, help="For a SQLite cache, directory for the .sqlite files containing the cache"
    )
    parser.add_argument(
        "--mongo-uri",
        type=str,
        help=(
            "For a MongoDB cache, Mongo URI to copy items to. "
            "Example format: mongodb://[username:password@]host1[:port1]/dbname"
        ),
    )
    parser.add_argument(
        "organization", type=str, help="Organization to export requests for", choices=["microsoft", "together"]
    )
    parser.add_argument("request_results_path", type=str, help="Path to jsonl file with requests and results.")
    parser.add_argument(
        "-d",
        "--dry-run",
        action="store_true",
        default=None,
        help="Skips persisting changes to sqlite file and prints log messages instead.",
    )
    args = parser.parse_args()

    if (args.cache_dir and args.mongo_uri) or (not args.cache_dir and not args.mongo_uri):
        raise ValueError("Exactly one of --cache-dir or --mongo-uri should be specified")
    cache_config: KeyValueStoreCacheConfig
    if args.cache_dir:
        cache_config = SqliteCacheConfig(os.path.join(args.cache_dir, f"{args.organization}.sqlite"))
    elif args.mongo_uri:
        cache_config = MongoCacheConfig(args.mongo_uri, args.organization)

    import_results(cache_config, args.organization, args.request_results_path, args.dry_run)
    hlog("Done.")
