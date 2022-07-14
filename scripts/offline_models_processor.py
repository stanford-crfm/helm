import argparse
import json
import os
from dacite import from_dict
from typing import Dict, List

from sqlitedict import SqliteDict

from common.request import Request
from common.cache import request_to_key
from common.hierarchical_logger import hlog, htrack
from proxy.together_client import TogetherClient


"""
Script to generate jsonl file with queries or update cache with responses for offline models (together/{model engine}).

Usage:

    python3 scripts/offline_models_processor.py <Command: either "generate" or "upload"> <Output path>

    Examples:
        python3 scripts/offline_models_processor.py generate benchmark_output/runs/dryrun --output-path queries.jsonl
        python3 scripts/offline_models_processor.py upload prod_env/cache/together.sqlite queries.jsonl responses.jsonl
"""


@htrack("Updating cache with results")
def upload_results(cache_path: str, queries_path: str, responses_path: str, dry_run: bool):
    """
    Given a jsonl file with queries and another jsonl file with results/responses, uploads
    query/response pairs to the cache at `cache_path`.
    """

    def read_jsonl(path: str) -> List[Dict]:
        result: List[Dict] = []
        with open(path, "r") as f:
            for line in f:
                result.append(json.loads(line))
        return result

    queries: List[Dict] = read_jsonl(queries_path)
    responses: List[Dict] = read_jsonl(responses_path)
    assert len(queries) == len(responses)

    with SqliteDict(cache_path) as cache:
        for query, response in zip(queries, responses):
            key = request_to_key(query)
            cache[key] = response

        if not dry_run:
            # Write to SQLite
            cache.commit()
            hlog(f"Wrote to {cache_path}.")


@htrack("Generating jsonl file with list of raw requests/queries")
def generate_queries(run_suite_path: str, output_path: str):
    """
    Given a run folder, generates a jsonl file at `output_path` with raw queries
    where each line represents a single request.
    """
    queries: List[str] = []

    for run_dir in os.listdir(run_suite_path):
        run_path: str = os.path.join(run_suite_path, run_dir)

        if not os.path.isdir(run_path):
            continue

        scenario_state_path: str = os.path.join(run_path, "scenario_state.json")
        if not os.path.isfile(scenario_state_path):
            continue

        with open(scenario_state_path) as f:
            scenario_state = json.load(f)
            model: str = scenario_state["adapter_spec"]["model"]
            assert model.startswith("together/")

            for request_state in scenario_state["request_states"]:
                request: Request = from_dict(Request, request_state["request"])
                raw_request = TogetherClient.convert_to_raw_request(request)
                queries.append(request_to_key(raw_request))

    with open(output_path, "w") as f:
        for i, query in enumerate(queries):
            f.write(query + "\n" if i < len(queries) - 1 else query)
        hlog(f"Wrote out {len(queries)} to {output_path}.")


def main():
    hlog(args)
    if args.command == "generate":
        generate_queries(args.run_path, args.output_path)
    elif args.command == "upload":
        upload_results(args.cache_path, args.queries_path, args.responses_path, args.dry_run)
    else:
        raise ValueError(f"Invalid command: {args.command}.")
    hlog("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    # Generate args
    generate_parser = subparsers.add_parser("generate", help="Generates queries.")
    generate_parser.add_argument("run_path", type=str, help="Path to run path.")
    generate_parser.add_argument("--output-path", type=str, default="queries.jsonl", help="Path to jsonl file.")

    # Upload args
    upload_parser = subparsers.add_parser("upload", help="Updates cache with queries and results.")
    upload_parser.add_argument("cache_path", type=str, help="Path to cache.")
    upload_parser.add_argument("queries_path", type=str, help="Path to jsonl file with queries.")
    upload_parser.add_argument("responses_path", type=str, help="Path to jsonl file with responses.")
    upload_parser.add_argument(
        "-d",
        "--dry-run",
        action="store_true",
        default=None,
        help="Skips persisting changes to sqlite file and prints log messages instead.",
    )

    args = parser.parse_args()

    main()
