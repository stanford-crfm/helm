"""Removes judge explanations on annotations

This script modifies all scenario_state.json files in place within a suite to
removing all explanations from judges from the `ScenarioState`s.

This is used when the scenario contains LLM as judge scores with score explanations
that may contain information about the dataset itself.

After running this, you must re-run helm-summarize on the suite in order to
update other JSON files used by the web frontend."""

import argparse
import dataclasses
import os
from typing import Dict, Any
from tqdm import tqdm

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.adaptation.scenario_state import ScenarioState
from helm.common.codec import from_json, to_json
from helm.common.hierarchical_logger import hwarn


SCENARIO_STATE_FILE_NAME = "scenario_state.json"
REDACTED_STRING = "[redacted]"


def read_scenario_state(scenario_state_path: str) -> ScenarioState:
    if not os.path.exists(scenario_state_path):
        raise ValueError(f"Could not load ScenarioState from {scenario_state_path}")
    with open(scenario_state_path) as f:
        return from_json(f.read(), ScenarioState)


def write_scenario_state(scenario_state_path: str, scenario_state: ScenarioState) -> None:
    with open(scenario_state_path, "w") as f:
        f.write(to_json(scenario_state))


def redact_annotations(annotations: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, val in annotations.items():
        if isinstance(val, dict):
            result[key] = redact_annotations(val)
        elif isinstance(val, str):
            if key in ["prompt_text", "explanation"]:
                result[key] = REDACTED_STRING
            else:
                result[key] = val
        else:
            result[key] = val

    return result


def redact_request_state(request_state: RequestState) -> RequestState:
    if request_state.annotations is None:
        return request_state
    annotations = redact_annotations(request_state.annotations)
    return dataclasses.replace(
        request_state,
        annotations=annotations,
    )


def redact_scenario_state(scenario_state: ScenarioState) -> ScenarioState:
    redacted_request_states = [redact_request_state(request_state) for request_state in scenario_state.request_states]
    return dataclasses.replace(scenario_state, request_states=redacted_request_states)


def modify_scenario_state_for_run(run_path: str) -> None:
    scenario_state_path = os.path.join(run_path, SCENARIO_STATE_FILE_NAME)
    scenario_state = read_scenario_state(scenario_state_path)
    redacted_scenario_state = redact_scenario_state(scenario_state)
    write_scenario_state(scenario_state_path, redacted_scenario_state)


def modify_scenario_states_for_suite(run_suite_path: str) -> None:
    """Load the runs in the run suite path."""
    # run_suite_path can contain subdirectories that are not runs (e.g. eval_cache, groups)
    # so filter them out.
    run_dir_names = sorted(
        [
            p
            for p in os.listdir(run_suite_path)
            if p != "eval_cache" and p != "groups" and os.path.isdir(os.path.join(run_suite_path, p))
        ]
    )
    for run_dir_name in tqdm(run_dir_names, disable=None):
        scenario_state_path: str = os.path.join(run_suite_path, run_dir_name, SCENARIO_STATE_FILE_NAME)
        if not os.path.exists(scenario_state_path):
            hwarn(f"{run_dir_name} doesn't have {SCENARIO_STATE_FILE_NAME}, skipping")
            continue
        run_path: str = os.path.join(run_suite_path, run_dir_name)
        modify_scenario_state_for_run(run_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--output-path", type=str, help="Where the benchmarking output lives", default="benchmark_output"
    )
    parser.add_argument(
        "--suite",
        type=str,
        help="Name of the suite this summarization should go under.",
    )
    args = parser.parse_args()
    output_path = args.output_path
    suite = args.suite
    run_suite_path = os.path.join(output_path, "runs", suite)
    modify_scenario_states_for_suite(run_suite_path)


if __name__ == "__main__":
    main()
