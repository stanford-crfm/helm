import cattrs
import importlib_resources as resources
import json
from typing import List, Optional


from helm.benchmark.data_overlap.light_scenario import AllGroupOverlapStats, GroupOverlapStats

OVERLAP_STATS_PACKAGE: str = "helm.benchmark.static"
PILE_OVERLAP_STATS_FILENAME: str = "overlap_stats.jsonl"


def read_overlap_stats() -> List[AllGroupOverlapStats]:
    try:
        overlap_stats_path = resources.files(OVERLAP_STATS_PACKAGE).joinpath(PILE_OVERLAP_STATS_FILENAME)
        overlap_stats_jsons = open(overlap_stats_path, "r").readlines()
        overlap_stats_list = []
        for overlap_stats_json in overlap_stats_jsons:
            overlap_stats_dict = json.loads(overlap_stats_json)
            overlap_stats_list.append(cattrs.structure(overlap_stats_dict, AllGroupOverlapStats))
        return overlap_stats_list
    except Exception:
        return []


def get_group_overlap_stats(
    overlap_stats_list: List[AllGroupOverlapStats], model: str, group: str
) -> Optional[GroupOverlapStats]:
    for overlap_stats in overlap_stats_list:
        models = overlap_stats.models
        if model in models:
            group_overlap_stats_list = overlap_stats.group_overlap_stats_list
            for group_overlap_stats in group_overlap_stats_list:
                curr_group = group_overlap_stats.group
                if curr_group == group:
                    return group_overlap_stats
    return None
