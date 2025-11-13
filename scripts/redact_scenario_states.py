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
from typing import Any, Dict, Optional
from tqdm import tqdm

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.adaptation.scenario_state import ScenarioState
from helm.benchmark.scenarios.scenario import Input, Instance, Output, Reference
from helm.common.codec import from_json, to_json
from helm.common.hierarchical_logger import hwarn
from helm.common.request import Request, RequestResult, GeneratedOutput, Token


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
    redacted_output = Output(text=REDACTED_STRING)
    return dataclasses.replace(reference, output=redacted_output)


def redact_instance(instance: Instance) -> Instance:
    redacted_input = Input(text=REDACTED_STRING)
    redacted_references = [redact_reference(reference) for reference in instance.references]
    return dataclasses.replace(instance, input=redacted_input, references=redacted_references)


def redact_request(request: Request) -> Request:
    return dataclasses.replace(request, prompt=REDACTED_STRING, messages=None, multimodal_prompt=None)


def redact_output_mapping(output_mapping: Optional[Dict[str, str]]) -> Optional[Dict[str, str]]:
    if output_mapping is None:
        return None
    return {key: REDACTED_STRING for key in output_mapping}


def redact_token(token: Token) -> Token:
    return dataclasses.replace(token, text=REDACTED_STRING)


def redact_completion(completion: GeneratedOutput) -> GeneratedOutput:
    # Replacing tokens for empty list in case length of completion reveals information about the prompt
    return dataclasses.replace(completion, text=REDACTED_STRING, tokens=[], multimodal_content=None, thinking=None)


def redact_result(result: RequestResult) -> RequestResult:
    redacted_completions = [redact_completion(completion) for completion in result.completions]
    return dataclasses.replace(result, completions=redacted_completions)


def redact_request_state_annotations(annotation: Any) -> Any:
    if isinstance(annotation, dict):
        return {key: redact_request_state_annotations(value) for key, value in annotation.items()}
    if isinstance(annotation, list):
        return [redact_request_state_annotations(elem) for elem in annotation]
    elif isinstance(annotation, str):
        return REDACTED_STRING
    else:
        return annotation


def redact_request_state(request_state: RequestState, redact_output: bool, redact_annotations: bool) -> RequestState:
    assert request_state.result is not None
    result = redact_result(request_state.result) if redact_output else request_state.result
    annotations = (
        redact_request_state_annotations(request_state.annotations)
        if redact_annotations and request_state.annotations
        else request_state.annotations
    )
    return dataclasses.replace(
        request_state,
        instance=redact_instance(request_state.instance),
        request=redact_request(request_state.request),
        output_mapping=redact_output_mapping(request_state.output_mapping),
        result=result,
        annotations=annotations,
    )


def redact_scenario_state(
    scenario_state: ScenarioState, redact_output: bool, redact_annotations: bool
) -> ScenarioState:
    redacted_request_states = [
        redact_request_state(request_state, redact_output, redact_annotations)
        for request_state in scenario_state.request_states
    ]
    return dataclasses.replace(scenario_state, request_states=redacted_request_states)


def modify_scenario_state_for_run(run_path: str, redact_output: bool, redact_annotations: bool) -> None:
    scenario_state_path = os.path.join(run_path, SCENARIO_STATE_FILE_NAME)
    scenario_state = read_scenario_state(scenario_state_path)
    redacted_scenario_state = redact_scenario_state(scenario_state, redact_output, redact_annotations)
    write_scenario_state(scenario_state_path, redacted_scenario_state)


def modify_scenario_states_for_suite(run_suite_path: str, redact_output: bool, redact_annotations: bool) -> None:
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
        modify_scenario_state_for_run(run_path, redact_output, redact_annotations)


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
    parser.add_argument("--redact-output", action="store_true", help="Whether to redact the generated outputs.")
    parser.add_argument("--redact-annotations", action="store_true", help="Whether to redact annotations.")
    args = parser.parse_args()
    output_path = args.output_path
    suite = args.suite
    redact_output = args.redact_output
    redact_annotations = args.redact_annotations
    run_suite_path = os.path.join(output_path, "runs", suite)
    modify_scenario_states_for_suite(run_suite_path, redact_output, redact_annotations)


if __name__ == "__main__":
    main()
