import argparse
import json
import cattrs
import pandas as pd
from enum import Enum
import numpy
from nltk import ngrams
from collections import defaultdict
from typing import List, Tuple, Any
from dataclasses import dataclass

from data_overlap_spec import DataOverlapStats, DataOverlapStatsKey, EntryOverlapNgrams
from compute_data_overlap_metrics import load_light_scenarios_from_jsonl
from common.util import get_tokenizer
from common.general import asdict_without_nones

@dataclass(frozen=True)
class EntryDataOverlapKey:
    """Unique key representing either the input or references of a single instance in a scenario."""

    stats_key: DataOverlapStatsKey
    part: str
    """Either PART_INPUT or PART_REF"""
    instance_id: str


# Input: List[EntryOverlapNgrams]
@dataclass(frozen=True)
class EntryOverlapNgrams:
    """Dataclass that represents output data overlap stats"""

    entry_data_overlap_key: EntryDataOverlapKey

    overlapping_ngram_counts: List[Tuple[str, int]]


class PartialOverlapSpec(int, Enum):
    binary = 0
    jaccard = 1
    token = 2
    def __str__(self):
        return self.name

@dataclass(frozen=True)
class FrequencySpec:
    # Filter ngrams with frequency >= filter_value; 0 means no filter
    filter_value: int
    # Whether to apply weight; we'll do inverse frequency
    weighting: bool
        
@dataclass(frozen=True)
class MetricProtocolSpec:
    """Specification for how we compute the metric"""
    
    partial_overlap_spec: PartialOverlapSpec
    frequency_spec: FrequencySpec
        
@dataclass(frozen=True)
class OverlapMetric:
    metric_score: float # use 0/1 for binary, can revise as neded
    metric_protocol_spec: MetricProtocolSpec

# Output: List[EntryOverlapMetric]
@dataclass(frozen=True)
class EntryOverlapMetric:
    """Dataclass that represents output data overlap stats"""

    entry_data_overlap_key: EntryDataOverlapKey

    overlap_metric: OverlapMetric

def scenario_spec_to_class(scenario_spec) -> str:
    return f"{'.'.join(scenario_spec.class_name.split('.')[-1:])}"     

PART_INPUT: str = "input"
PART_REF: str = "reference"
metric_protocol_specs_list  = [
    MetricProtocolSpec(PartialOverlapSpec.binary, FrequencySpec(0, False)),
    MetricProtocolSpec(PartialOverlapSpec.jaccard, FrequencySpec(0, False)),
    MetricProtocolSpec(PartialOverlapSpec.jaccard, FrequencySpec(0, True)),
    MetricProtocolSpec(PartialOverlapSpec.token, FrequencySpec(0, False)),
    MetricProtocolSpec(PartialOverlapSpec.token, FrequencySpec(0, True)),
    MetricProtocolSpec(PartialOverlapSpec.binary, FrequencySpec(10, False)),
    MetricProtocolSpec(PartialOverlapSpec.jaccard, FrequencySpec(10, False)),
    MetricProtocolSpec(PartialOverlapSpec.jaccard, FrequencySpec(10, True)),
    MetricProtocolSpec(PartialOverlapSpec.token, FrequencySpec(10, False)),
    MetricProtocolSpec(PartialOverlapSpec.token, FrequencySpec(10, True))
]

@dataclass(frozen=True)
class AggregateDataOverlapKey:
    """Key representing the aggregated data overlap stats"""
    stats_key: DataOverlapStatsKey
    part: str

@dataclass(frozen=True)
class AggregateOverlapMetric:
    """Dataclass representing the aggregated overlap metrics"""
    aggregate_data_overlap_key: AggregateDataOverlapKey
    metric_scores: List[float]  # List of scores instead of a single value
    metric_protocol_spec: MetricProtocolSpec

def aggregate_metrics(path, out_path):
    overlap_metrics_jsons = open(path, "r").readlines()

    entry_overlap_metric_list = []
    for entry_overlap_metric_json in overlap_metrics_jsons:
        entry_overlap_metric_dict = json.loads(entry_overlap_metric_json)
        entry_overlap_metric_list.append(cattrs.structure(entry_overlap_metric_dict, EntryOverlapMetric))

    # Initialize a new dictionary for aggregated scores
    aggregate_score_dict = {}

    for entry_overlap_metric in entry_overlap_metric_list:
        # Extract necessary information
        stats_key = entry_overlap_metric.entry_data_overlap_key.stats_key
        part = entry_overlap_metric.entry_data_overlap_key.part
        metric_protocol_spec = entry_overlap_metric.overlap_metric.metric_protocol_spec
        metric_score = entry_overlap_metric.overlap_metric.metric_score

        # Define the aggregate key
        agg_key = (stats_key, part, metric_protocol_spec)

        # Initialize or append the metric score
        if agg_key not in aggregate_score_dict:
            aggregate_score_dict[agg_key] = []
        aggregate_score_dict[agg_key].append(metric_score)

    # Convert the aggregated data to AggregateOverlapMetric objects
    aggregate_overlap_metrics = []
    for (stats_key, part, metric_protocol_spec), scores in aggregate_score_dict.items():
        aggregate_key = AggregateDataOverlapKey(
            stats_key=stats_key,
            part=part
        )
        aggregate_overlap_metrics.append(
            AggregateOverlapMetric(
                aggregate_data_overlap_key=aggregate_key,
                metric_scores=scores,
                metric_protocol_spec=metric_protocol_spec
            )
        )

    def save_metrics_to_jsonl(overlap_metrics: List[AggregateOverlapMetric], filename: str):
        with open(filename, "w") as f:
            for overlap_metric in overlap_metrics:
                f.write(json.dumps(asdict_without_nones(overlap_metric), ensure_ascii=False) + "\n")
    
    save_metrics_to_jsonl(aggregate_overlap_metrics, out_path)

def get_args() -> Any:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-path", type=str, required=True, help="Path to your metrics")
    parser.add_argument("--out-path", type=str, required=True, help="Path to the output metrics file")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    aggregate_metrics(args.metrics_path, args.out_path)
