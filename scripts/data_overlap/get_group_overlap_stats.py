import json
import cattrs
import os
import argparse
from typing import List, Dict

from common.general import asdict_without_nones

from data_overlap_spec import DataOverlapStats
from light_scenario import GroupScenarioSpecs, GroupOverlapStats


def save_group_overlap_stats_to_jsonl(group_overlap_stats_list: List[GroupOverlapStats], filename: str):
    with open(filename, "a") as f:
        for group_overlap_stats in group_overlap_stats_list:
            f.write(json.dumps(asdict_without_nones(group_overlap_stats), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--overlap_stats_path", type=str, required=True, help="The path to the overlap stats file")
    parser.add_argument(
        "--group_scenario_specs_path", type=str, required=True, help="The path to the group scenario specs file"
    )
    parser.add_argument("--output-data", type=str, required=True, help="The path to the output file")
    args = parser.parse_args()

    try:
        os.remove(args.output_data)
    except OSError:
        pass

    overlap_stats_jsons = open(args.overlap_stats_path, "r").readlines()

    data_overlap_stats_list = []
    for overlap_stats_json in overlap_stats_jsons:
        overlap_stats_dict = json.loads(overlap_stats_json)
        data_overlap_stats_list.append(cattrs.structure(overlap_stats_dict, DataOverlapStats))

    scenario_spec_overlap_counts: Dict = dict()
    for data_overlap_stats in data_overlap_stats_list:
        data_overlap_stats_key = data_overlap_stats.data_overlap_stats_key
        light_scenario_key = data_overlap_stats_key.light_scenario_key
        scenario_spec = light_scenario_key.scenario_spec
        num_instances = data_overlap_stats.num_instances
        n = data_overlap_stats_key.overlap_protocol_spec.n
        num_overlapping_inputs = len(data_overlap_stats.instance_ids_with_overlapping_input)
        num_overlapping_references = len(data_overlap_stats.instance_ids_with_overlapping_reference)
        if n == 13:
            scenario_spec_overlap_counts[scenario_spec] = (
                num_instances,
                num_overlapping_inputs,
                num_overlapping_references,
            )

    group_scenario_specs_jsons = open(args.group_scenario_specs_path, "r").readlines()

    group_scenario_specs_list: List = []
    for group_scenario_specs_json in group_scenario_specs_jsons:
        group_scenario_specs_dict = json.loads(group_scenario_specs_json)
        group_scenario_specs_list.append(cattrs.structure(group_scenario_specs_dict, GroupScenarioSpecs))

    group_overlap_stats_list: List = []
    for group_scenario_specs in group_scenario_specs_list:
        group = group_scenario_specs.group
        scenario_specs = group_scenario_specs.scenario_specs
        group_num_instances = 0
        group_num_overlapping_inputs = 0
        group_num_overlapping_references = 0
        for scenario_spec in scenario_specs:
            try:
                num_instances, num_overlapping_inputs, num_overlapping_references = scenario_spec_overlap_counts[
                    scenario_spec
                ]
                group_num_instances += num_instances
                group_num_overlapping_inputs += num_overlapping_inputs
                group_num_overlapping_references += num_overlapping_references
            except Exception:
                pass
        if group_num_instances != 0:
            group_overlap_stats = GroupOverlapStats(
                group=group,
                num_instances=group_num_instances,
                num_overlapping_inputs=group_num_overlapping_inputs,
                num_overlapping_references=group_num_overlapping_references,
            )
            group_overlap_stats_list.append(group_overlap_stats)

    save_group_overlap_stats_to_jsonl(group_overlap_stats_list, args.output_data)
