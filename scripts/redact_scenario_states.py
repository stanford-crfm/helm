"""Removes prompts from scenario state.

This script modifies all scenario_state.json files in place within a suite to
removing all prompts, instance input text, and instance reference output text
from the `ScenarioState`s.

This is used when the scenario contains prompts that should not be displayed,
in order to reduce the chance of data leakage or to comply with data privacy
requirements.

After running this, you must re-run helm-summarize on the suite in order to
update other JSON files used by the web frontend."""

import argparse
import dataclasses
import os
from typing import Dict, Optional
from tqdm import tqdm

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.adaptation.scenario_state import ScenarioState
from helm.benchmark.scenarios.scenario import Instance, Reference
from helm.common.codec import from_json, to_json
from helm.common.hierarchical_logger import hlog
from helm.common.request import Request


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


def redact_reference(reference: Reference) -> Reference:
    redacted_output = dataclasses.replace(reference.output, text=REDACTED_STRING)
    return dataclasses.replace(reference, output=redacted_output)


def redact_instance(instance: Instance) -> Instance:
    redacted_input = dataclasses.replace(instance.input, text=REDACTED_STRING)
    redacted_references = [redact_reference(reference) for reference in instance.references]
    return dataclasses.replace(instance, input=redacted_input, references=redacted_references)


def redact_request(request: Request) -> Request:
    return dataclasses.replace(request, prompt=REDACTED_STRING, messages=None, multimodal_prompt=None)


def redact_output_mapping(output_mapping: Optional[Dict[str, str]]) -> Optional[Dict[str, str]]:
    if output_mapping is None:
        return None
    return {key: REDACTED_STRING for key in output_mapping}


def redact_request_state(request_state: RequestState) -> RequestState:
    return dataclasses.replace(
        request_state,
        instance=redact_instance(request_state.instance),
        request=redact_request(request_state.request),
        output_mapping=redact_output_mapping(request_state.output_mapping),
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
            hlog(f"WARNING: {run_dir_name} doesn't have {SCENARIO_STATE_FILE_NAME}, skipping")
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
