import json
import os
import argparse
from typing import List
from collections import defaultdict

from helm.common.general import asdict_without_nones, ensure_directory_exists
from helm.common.hierarchical_logger import hlog, htrack_block

from helm.benchmark.scenarios.scenario import Scenario, create_scenario, Instance
from helm.benchmark.presentation.run_entry import read_run_entries
from helm.benchmark.run import run_entries_to_run_specs
from helm.benchmark.runner import RunSpec
from helm.benchmark.compute_contamination_metrics import LightInstance, LightScenario


def create_light_instance_from_helm_instance(instance: Instance) -> LightInstance:
    """Create a LightInstance given a helm Instance. Only keep the text attributes."""
    input_text: str = instance.input.text
    reference_texts: List[str] = [reference.output.text for reference in instance.references]
    return LightInstance(input=input_text, references=reference_texts)


def get_light_scenarios_from_helm(
    run_spec: RunSpec, scenario_download_path: str = "exported_scenarios"
) -> List[LightScenario]:
    """Create a list of LightInstances given a helm run_spec. Only keep the text attributes."""

    scenario: Scenario = create_scenario(run_spec.scenario_spec)

    ensure_directory_exists(scenario_download_path)
    scenario.output_path = os.path.join(scenario_download_path, scenario.name)
    ensure_directory_exists(scenario.output_path)

    scenario_name = scenario.name
    if len(run_spec.scenario_spec.args) > 0:
        scenario_name += f",{','.join([key+'='+value for key, value in run_spec.scenario_spec.args.items()])}"

    # Load instances
    helm_instances: List[Instance]
    with htrack_block("scenario.get_instances"):
        helm_instances = scenario.get_instances()

    # Classify instances into splits
    splits = defaultdict(list)
    for instance in helm_instances:
        splits[instance.split].append(instance)

    light_scenarios: List[LightScenario] = []
    for split, instances in splits.items():
        light_instances: List[LightInstance] = [
            create_light_instance_from_helm_instance(instance) for instance in instances
        ]
        light_scenario = LightScenario(name=scenario_name + f",split={split}", light_instances=light_instances)
        light_scenarios.append(light_scenario)
    return light_scenarios


def save_scenarios_to_jsonl(light_scenarios: List[LightScenario], filename: str):
    """
    Save a list of LightInstance to the jsonl format described below.

    Input File Format:

    Instance JSON 1
    Instance JSON 2
    Instance JSON 3
    ...

    Each line is a json and each json looks like:
    {
        "name": "SCENARIO_NAME",
        "light_instances": [
            {
            "input": "INPUT_TEXT1",
            "references": [
                "REFERENCE_TEXT_1",
                "REFERENCE_TEXT_2"
            ]
            },
            {
            "input": "INPUT_TEXT2",
            "references": [
                "REFERENCE_TEXT_3",
                "REFERENCE_TEXT_4"
            ]
            }
        ]
    }
    """
    with open(filename, "w") as f:
        for light_scenario in light_scenarios:
            f.write(json.dumps(asdict_without_nones(light_scenario), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-specs", nargs="+", required=True, help="Specifies what to export")
    parser.add_argument("--output-data", type=str, required=True, help="The path to the output file")
    args = parser.parse_args()

    hlog("Loading run_specs")
    run_entries = read_run_entries(args.run_specs).entries
    run_specs = run_entries_to_run_specs(
        run_entries=run_entries,
    )

    hlog("Loading instances from helm scenarios")
    light_scenarios: List[LightScenario] = []
    for run_spec in run_specs:
        light_scenarios.extend(get_light_scenarios_from_helm(run_spec))

    save_scenarios_to_jsonl(light_scenarios, args.output_data)
