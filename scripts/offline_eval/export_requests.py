import argparse
import json
import os
import typing
from collections import Counter
from dacite import from_dict

from helm.common.request import Request
from helm.common.cache import (
    KeyValueStoreCacheConfig,
    MongoCacheConfig,
    SqliteCacheConfig,
    create_key_value_store,
    request_to_key,
)
from helm.common.hierarchical_logger import hlog, htrack, htrack_block
from helm.proxy.clients.together_client import TogetherClient
from helm.proxy.clients.microsoft_client import MicrosoftClient


"""
Exports raw requests from a run suite to a jsonl file.

Usage:

    python3 scripts/offline_eval/export_requests.py <Org - one of 'microsoft' or 'together'> <Path to run suite>

    Example:

      python3 scripts/offline_eval/export_requests.py together benchmark_output/runs/v4-dryrun

"""


@htrack("Generating jsonl file with list of raw requests")
def export_requests(cache_config: KeyValueStoreCacheConfig, organization: str, run_suite_path: str, output_path: str):
    """
    Given a run suite folder, generates a jsonl file at `output_path` with raw queries
    where each line represents a single request.
    """

    def process_together_request(request: Request):
        raw_request: typing.Dict = TogetherClient.convert_to_raw_request(request)
        # Only export requests that we are not in the cache
        if not store.contains(raw_request):
            request_json: str = request_to_key(raw_request)
            out_file.write(request_json + "\n")
            counts["pending_count"] += 1
        else:
            counts["cached_count"] += 1

    def process_microsoft_request(request: Request):
        raw_request: typing.Dict = MicrosoftClient.convert_to_raw_request(request)
        for completion_index in range(request.num_completions):
            # We send the same request `num_completions` times because the MT-NLG API does not
            # support the OpenAI parameter 'n'. In our cache, we use `completion_index` to
            # differentiate responses for the same request, so we should check if the
            # request + 'completion_index` is in our cache. However, when we write out the
            # requests for offline batch evaluation, we should exclude `completion_index`
            # and write out the JSON for the same request `num_completion` times.
            cache_key: typing.Dict = {"completion_index": completion_index, **raw_request}

            # Only export requests that we are not in the cache
            if not store.contains(cache_key):
                request_json: str = request_to_key(raw_request)
                out_file.write(request_json + "\n")
                counts["pending_count"] += 1
            else:
                counts["cached_count"] += 1

    counts: typing.Counter = Counter(pending_count=0, cached_count=0)

    # Go through all the valid run folders, pull requests from the scenario_state.json files
    # and write them out to the jsonl file at path `output_path`.
    with create_key_value_store(cache_config) as store:
        with open(output_path, "w") as out_file:
            for run_dir in os.listdir(run_suite_path):
                run_path: str = os.path.join(run_suite_path, run_dir)

                if not os.path.isdir(run_path):
                    continue

                with htrack_block(f"Processing run directory: {run_dir}"):
                    scenario_state_path: str = os.path.join(run_path, "scenario_state.json")
                    if not os.path.isfile(scenario_state_path):
                        hlog(
                            f"{run_dir} is missing a scenario_state.json file. Expected at path: {scenario_state_path}."
                        )
                        continue

                    with open(scenario_state_path) as scenario_state_file:
                        scenario_state = json.load(scenario_state_file)
                        model_name: str = scenario_state["adapter_spec"]["model"]
                        current_organization: str = model_name.split("/")[0]

                        if current_organization != organization:
                            hlog(f"Not generating requests for {current_organization}.")
                            continue

                        for request_state in scenario_state["request_states"]:
                            request: Request = from_dict(Request, request_state["request"])
                            if current_organization == "together":
                                process_together_request(request)
                            elif current_organization == "microsoft":
                                try:
                                    process_microsoft_request(request)
                                except ValueError as e:
                                    hlog(f"Error while processing Microsoft request: {e}\nRequest: {request}")
                            else:
                                raise ValueError(f"Unhandled organization: {current_organization}.")

                    hlog(f"Wrote {counts['pending_count']} requests so far.")

    hlog(
        f"Wrote {counts['pending_count']} requests to {output_path}. "
        f"{counts['cached_count']} requests already had an entry in the cache."
    )


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
    parser.add_argument("run_suite_path", type=str, help="Path to run path.")
    parser.add_argument("--output-path", type=str, default="requests.jsonl", help="Path to jsonl file.")
    args = parser.parse_args()

    if (args.cache_dir and args.mongo_uri) or (not args.cache_dir and not args.mongo_uri):
        raise ValueError("Exactly one of --cache-dir or --mongo-uri should be specified")
    cache_config: KeyValueStoreCacheConfig
    if args.cache_dir:
        cache_config = SqliteCacheConfig(os.path.join(args.cache_dir, f"{args.organization}.sqlite"))
    elif args.mongo_uri:
        cache_config = MongoCacheConfig(args.mongo_uri, args.organization)

    export_requests(cache_config, args.organization, args.run_suite_path, args.output_path)
    hlog("Done.")
