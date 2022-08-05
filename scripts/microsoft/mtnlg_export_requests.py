import argparse
import json
import os
from dacite import from_dict

from sqlitedict import SqliteDict

from common.request import Request
from common.cache import request_to_key
from common.hierarchical_logger import hlog, htrack, htrack_block
from proxy.clients.microsoft_client import MicrosoftClient


"""
Exports raw requests from a run suite to a jsonl file for the MT-NLG models (microsoft/{model engine}).

Usage:

  python3 scripts/microsoft/mtnlg_export_requests.py <Path to run suite> <Path to MicrosoftClient's cache>
 
  Example:

  python3 scripts/microsoft/mtnlg_export_requests.py benchmark_output/runs/microsoft prod_env/cache/microsoft.sqlite

"""


@htrack("Generating jsonl file with list of raw requests")
def export_requests(run_suite_path: str, cache_path: str, output_path: str):
    """
    Given a run suite folder, generates a jsonl file at `output_path` with raw queries
    where each line represents a single request.
    """
    count: int = 0
    cached_count: int = 0

    # Goes through all the valid run folders, pulls requests from the scenario_state.json files
    # and writes them out to the jsonl file at path `output_path`.
    with SqliteDict(cache_path) as cache:
        with open(output_path, "w") as out_file:
            for run_dir in os.listdir(run_suite_path):
                run_path: str = os.path.join(run_suite_path, run_dir)

                if not os.path.isdir(run_path):
                    continue

                with htrack_block(f"Processing run directory: {run_dir}"):
                    scenario_state_path: str = os.path.join(run_path, "scenario_state.json")
                    hlog(f"{run_dir} is missing a scenario_state.json file. Expected at path: {scenario_state_path}.")
                    if not os.path.isfile(scenario_state_path):
                        continue

                    with open(scenario_state_path) as scenario_state_file:
                        scenario_state = json.load(scenario_state_file)
                        model_name: str = scenario_state["adapter_spec"]["model"]
                        organization: str = model_name.split("/")[0]

                        # We only need to export raw queries for MT-NLG models
                        if organization != "microsoft":
                            continue

                        for request_state in scenario_state["request_states"]:
                            request: Request = from_dict(Request, request_state["request"])
                            for completion_index in range(request.num_completions):
                                raw_request = MicrosoftClient.convert_to_raw_request(request)

                                # We send the same request `num_completions` times because the MT-NLG API does not
                                # support the OpenAI parameter 'n'. In our cache, we use `completion_index` to
                                # differentiate responses for the same request, so we should check if the
                                # request + 'completion_index` is in our cache. However, when we write out the
                                # requests for offline batch evaluation, we should exclude `completion_index`
                                # and write out the JSON for the same request `num_completion` times.
                                cache_key: str = request_to_key({"completion_index": completion_index, **raw_request})
                                request_json: str = request_to_key(raw_request)

                                # Only export requests that we are not in the cache
                                if cache_key not in cache:
                                    out_file.write(request_json + "\n")
                                    count += 1
                                else:
                                    cached_count += 1
                    hlog(f"Wrote {count} requests so far.")

    hlog(f"Wrote {count} requests to {output_path}. {cached_count} requests already had an entry in the cache.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_suite_path", type=str, help="Path to run path.")
    parser.add_argument(
        "cache_path", type=str, help="Path to sqlite file (microsoft.sqlite) for the `MicrosoftClient` cache."
    )
    parser.add_argument("--output-path", type=str, default="mtnlg_requests.jsonl", help="Path to jsonl file.")
    args = parser.parse_args()

    export_requests(args.run_suite_path, args.cache_path, args.output_path)
    hlog("Done.")
