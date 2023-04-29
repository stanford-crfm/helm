import json
import os
import argparse
from typing import List, DefaultDict
from collections import defaultdict

from helm.common.general import asdict_without_nones, ensure_directory_exists
from helm.common.hierarchical_logger import hlog, htrack_block

from helm.benchmark.scenarios.scenario import Scenario, Instance, create_scenario, TRAIN_SPLIT, VALID_SPLIT, TEST_SPLIT
from helm.benchmark.presentation.run_entry import read_run_entries
from helm.benchmark.run import run_entries_to_run_specs
from helm.benchmark.runner import RunSpec
from helm.benchmark.contamination.light_scenario import LightInstance, LightScenario, LightScenarioKey


def create_light_instance_from_instance(instance: Instance) -> LightInstance:
    """Create a LightInstance given an Instance. Only keep the text attributes."""
    input_text: str = instance.input.text
    reference_texts: List[str] = [reference.output.text for reference in instance.references]
    return LightInstance(input=input_text, references=reference_texts)


def get_light_scenarios_from_run_spec(
    run_spec: RunSpec, scenario_download_path: str = "exported_scenarios"
) -> List[LightScenario]:
    """
    Create a list of LightInstances given a RunSpec. Only keep the text of the input and references.
    Note that one LightScenario object is created for each split of the Scenario for simplification.
    """

    scenario: Scenario = create_scenario(run_spec.scenario_spec)

    ensure_directory_exists(scenario_download_path)
    scenario.output_path = os.path.join(scenario_download_path, scenario.name)
    ensure_directory_exists(scenario.output_path)

    # Load instances
    instances: List[Instance]
    with htrack_block("scenario.get_instances"):
        instances = scenario.get_instances()

    # Classify instances into splits
    splits: List[str] = [TRAIN_SPLIT, VALID_SPLIT, TEST_SPLIT]
    split_mapping: DefaultDict[str, list] = defaultdict(list)
    for instance in instances:
        if instance.split is None or instance.split not in splits:
            raise ValueError(
                f"split should be one of {TRAIN_SPLIT}, {VALID_SPLIT}, or {TEST_SPLIT}, but got {instance.split}"
            )
        split_mapping[instance.split].append(instance)

    # Convert Scenarios to LightScenarios
    light_scenarios: List[LightScenario] = []
    for split, instances in split_mapping.items():
        light_instances: List[LightInstance] = [create_light_instance_from_instance(instance) for instance in instances]
        light_scenario = LightScenario(
            light_scenario_key=LightScenarioKey(metadata={"scenario_spec": run_spec.scenario_spec, "split": split}),
            light_instances=light_instances,
        )
        light_scenarios.append(light_scenario)
    return light_scenarios


def save_scenarios_to_jsonl(light_scenarios: List[LightScenario], filename: str):
    """
    Save a list of LightInstance to a jsonl file where each line represents a LightScenario object.
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

    hlog("Generating light scenarios from scenarios")
    light_scenarios: List[LightScenario] = []
    for run_spec in run_specs:
        light_scenarios.extend(get_light_scenarios_from_run_spec(run_spec))

    save_scenarios_to_jsonl(light_scenarios, args.output_data)
