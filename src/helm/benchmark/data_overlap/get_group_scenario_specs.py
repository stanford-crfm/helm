import json
import os
import argparse
from typing import List, DefaultDict, Set, Dict
from collections import defaultdict

from helm.common.general import asdict_without_nones
from helm.common.hierarchical_logger import hlog


from helm.benchmark.presentation.run_entry import read_run_entries
from helm.benchmark.run import run_entries_to_run_specs
from helm.benchmark.data_overlap.light_scenario import  GroupScenarioSpecs



def save_group_scenario_specs_to_jsonl(group_scenario_specs: List[GroupScenarioSpecs], filename: str):
    with open(filename, "a") as f:
        for scenario_spec_groups in group_scenario_specs:
            f.write(json.dumps(asdict_without_nones(scenario_spec_groups), ensure_ascii=False) + "\n")


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
    scenario_specs_to_groups: Dict = dict()
    for run_spec in run_specs:
        scenario_spec = run_spec.scenario_spec
        groups = run_spec.groups
        if (
            scenario_spec.class_name
            != "helm.benchmark.scenarios.synthetic_efficiency_scenario.SyntheticEfficiencyScenario"
        ):
            scenario_specs.add(scenario_spec)
            scenario_specs_to_groups[scenario_spec] = groups
    
    group_to_scenario_specs: DefaultDict = defaultdict(list)
    for scenario_spec, groups in scenario_specs_to_groups.items():
        for group in groups:
            group_to_scenario_specs[group].append(scenario_spec)


    group_scenario_specs: List = []
    for group, scenario_specs in group_to_scenario_specs.items():
        group_scenario_specs.append(GroupScenarioSpecs( group=group, scenario_specs=scenario_spec))

    save_group_scenario_specs_to_jsonl( group_scenario_specs, args.output_data)
