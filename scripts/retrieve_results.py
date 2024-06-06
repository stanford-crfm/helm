import os
import requests
import argparse

from helm.benchmark.adaptation.scenario_state import ScenarioState
from helm.benchmark.run_spec import RunSpec
from helm.common.codec import from_json, to_json
from typing import List


BASE_URL = "https://storage.googleapis.com/crfm-helm-public/"
DEFAULT_PROJECT_ID = "medical"
DEFAULT_PROJECT_RELEASE = "v0.1.0"

# TODO support reading from local benchmark_output directory


def get_model_from_run_name(run_name: str) -> str:
    candidate = run_name.split(":")[1].replace("model=", " ")
    if "," in candidate:
        splitted = candidate.split(",")
        if "stop=" in candidate:
            return splitted[len(splitted) - 3]
        else:
            return splitted[len(splitted) - 2]
    else:
        return candidate


def get_scenario_from_run_name(run_name: str) -> str:
    return run_name.split(":")[0]


def get_scenario_details_from_run_name(run_name: str) -> str:
    scenario_name = run_name.split(":")[0]
    candidate = run_name.split(":")[1].split(",")


def get_run_specs(project_id: str, release_id: str) -> list:
    response = requests.get(f"{BASE_URL}{project_id}/benchmark_output/releases/{release_id}/run_specs.json")
    if response.status_code != 200:
        raise ValueError(
            f"Could not load run names from {project_id}/{release_id}, status code: {response.status_code}"
        )
    return from_json(response.text, List[RunSpec])


def read_scenario_state(scenario_state_url: str) -> ScenarioState:
    response = requests.get(scenario_state_url)
    if response.status_code != 200:
        raise ValueError(f"Could not load ScenarioState from {scenario_state_url}, status code: {response.status_code}")
    return from_json(response.text, ScenarioState)


def create_scenario_state_url(run_name: str, project_id: str, release_id: str) -> str:
    return f"{BASE_URL}{project_id}/benchmark_output/runs/{release_id}/{run_name}/scenario_state.json"


def main():
    parser = argparse.ArgumentParser(description="Retrieve results from a HELM run")
    parser.add_argument("--project_id", type=str, default=DEFAULT_PROJECT_ID, help="The project ID")
    parser.add_argument("--release_id", type=str, default=DEFAULT_PROJECT_RELEASE, help="The release ID")

    args = parser.parse_args()

    run_specs = get_run_specs(args.project_id, args.release_id)
    run_names = []
    for r in run_specs:
        run_names.append(r.name)

    all_scenario_states = {}

    for count, r in enumerate(run_names, start=1):
        print(f"Iteration {count} / {len(run_names)}: Processing run name '{r}'")
        scenario_state_url = create_scenario_state_url(r, args.project_id, args.release_id)
        scenario_state = read_scenario_state(scenario_state_url)
        all_scenario_states[r] = scenario_state


if __name__ == "__main__":
    main()
