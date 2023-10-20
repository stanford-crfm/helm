import json
import os
import argparse
from typing import List, DefaultDict, Set
from collections import defaultdict

from helm.common.general import asdict_without_nones, ensure_directory_exists
from helm.common.hierarchical_logger import hlog, htrack_block

from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    create_scenario,
    TRAIN_SPLIT,
    VALID_SPLIT,
    TEST_SPLIT,
    ScenarioSpec,
    with_instance_ids,
)
from helm.benchmark.presentation.run_entry import read_run_entries
from helm.benchmark.run import run_entries_to_run_specs
from helm.benchmark.data_overlap.light_scenario import LightInstance, LightScenario, LightScenarioKey


def create_light_instance_from_instance(instance: Instance) -> LightInstance:
    """Create a LightInstance given an Instance. Only keep the text attributes."""
    input_text: str = instance.input.text
    reference_texts: List[str] = [reference.output.text for reference in instance.references]
    return LightInstance(input=input_text, references=reference_texts, id=instance.id)


def get_light_scenarios_from_scenario_spec(
    scenario_spec: ScenarioSpec, scenario_download_path: str = "exported_scenarios"
) -> List[LightScenario]:
    """
    Create a list of LightInstances given a ScenarioSpec. Only keep the text of the input and references.
    Note that one LightScenario object is created for each split of the Scenario for simplification.
    """

    scenario: Scenario = create_scenario(scenario_spec)

    ensure_directory_exists(scenario_download_path)
    scenario_output_path = os.path.join(scenario_download_path, scenario.name)
    ensure_directory_exists(scenario_output_path)

    # Load instances
    instances: List[Instance]
    with htrack_block("scenario.get_instances"):
        instances = scenario.get_instances(scenario_output_path)

    # Get instance ids
    instances = with_instance_ids(instances)

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
        light_scenario_key: LightScenarioKey = LightScenarioKey(
            scenario_spec=scenario_spec,
            split=split,
        )
        light_scenario = LightScenario(
            scenario_key=light_scenario_key,
            instances=light_instances,
        )
        light_scenarios.append(light_scenario)
    return light_scenarios


def save_scenarios_to_jsonl(light_scenarios: List[LightScenario], filename: str):
    """
    Save a list of LightInstance to a jsonl file where each line represents a LightScenario object.
    """
    with open(filename, "a") as f:
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
        priority=4,
    )

    try:
        os.remove(args.output_data)
    except OSError:
        pass

    scenario_specs: Set = set()
    for run_spec in run_specs:
        scenario_spec = run_spec.scenario_spec
        if (
            scenario_spec.class_name
            != "helm.benchmark.scenarios.synthetic_efficiency_scenario.SyntheticEfficiencyScenario"
        ):
            scenario_specs.add(scenario_spec)

    hlog("Generating light scenarios from scenarios")
    for scenario_spec in scenario_specs:
        light_scenarios: List[LightScenario] = get_light_scenarios_from_scenario_spec(scenario_spec)
        save_scenarios_to_jsonl(light_scenarios, args.output_data)
