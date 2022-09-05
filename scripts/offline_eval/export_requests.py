import argparse
import json
import os
from dacite import from_dict

from sqlitedict import SqliteDict

from common.request import Request
from common.cache import request_to_key
from common.hierarchical_logger import hlog, htrack, htrack_block
from proxy.clients.together_client import TogetherClient
from proxy.clients.microsoft_client import MicrosoftClient


"""
Exports raw requests from a run suite to a jsonl file.

Usage:

    python3 scripts/offline_eval/export_requests.py <Org - one of 'microsoft' or 'together'> <Path to run suite>

    Example:

      python3 scripts/offline_eval/export_requests.py together benchmark_output/runs/v4dryrun

"""


@htrack("Generating jsonl file with list of raw requests")
def export_requests(organization: str, run_suite_path: str, output_path: str):
    """
    Given a run suite folder, generates a jsonl file at `output_path` with raw queries
    where each line represents a single request.
    """
    pending_count: int = 0
    cached_count: int = 0

    # Go through all the valid run folders, pull requests from the scenario_state.json files
    # and write them out to the jsonl file at path `output_path`.
    with SqliteDict(f"prod_env/cache/{organization}.sqlite") as cache:
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
                            cache_key: str
                            request_json: str

                            if current_organization == "together":
                                raw_request = TogetherClient.convert_to_raw_request(request)
                                cache_key = request_to_key(raw_request)

                                # Only export requests that we are not in the cache
                                if cache_key not in cache:
                                    # Following the examples from https://github.com/togethercomputer/open-models-api,
                                    # add "request_type" and "model" to the request and remove "engine".
                                    raw_request.pop("engine", None)
                                    request_json = request_to_key(
                                        {
                                            "request_type": "language-model-inference",
                                            "model": request.model,
                                            **raw_request,
                                        }
                                    )
                                    out_file.write(request_json + "\n")
                                    pending_count += 1
                                else:
                                    cached_count += 1
                            elif current_organization == "microsoft":
                                raw_request = MicrosoftClient.convert_to_raw_request(request)
                                for completion_index in range(request.num_completions):
                                    # We send the same request `num_completions` times because the MT-NLG API does not
                                    # support the OpenAI parameter 'n'. In our cache, we use `completion_index` to
                                    # differentiate responses for the same request, so we should check if the
                                    # request + 'completion_index` is in our cache. However, when we write out the
                                    # requests for offline batch evaluation, we should exclude `completion_index`
                                    # and write out the JSON for the same request `num_completion` times.
                                    cache_key = request_to_key({"completion_index": completion_index, **raw_request})

                                    # Only export requests that we are not in the cache
                                    if cache_key not in cache:
                                        request_json = request_to_key(raw_request)
                                        out_file.write(request_json + "\n")
                                        pending_count += 1
                                    else:
                                        cached_count += 1
                    hlog(f"Wrote {pending_count} requests so far.")

    hlog(f"Wrote {pending_count} requests to {output_path}. {cached_count} requests already had an entry in the cache.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("organization", type=str, help="Organization to export requests for", choices=["microsoft", "together"])
    parser.add_argument("run_suite_path", type=str, help="Path to run path.")
    parser.add_argument("--output-path", type=str, default="requests.jsonl", help="Path to jsonl file.")
    args = parser.parse_args()

    export_requests(args.organization, args.run_suite_path, args.output_path)
    hlog("Done.")
