
ngram_path =  'output_stats_pi..gram_xaa_ngrams'
scenario_path = './data/xaa'
out_path = 'metrics_xaa'

# Computing for N = 13 for illustrative purposes
N = 13



import ast
import json
import cattrs
import pandas as pd
from nltk import ngrams
from collections import defaultdict
from typing import List, Tuple
from dataclasses import dataclass

from data_overlap_spec import DataOverlapStats, DataOverlapStatsKey, EntryOverlapNgrams
from compute_data_overlap_metrics import load_light_scenarios_from_jsonl
from common.util import get_tokenizer
from common.general import asdict_without_nones

from enum import Enum


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

def get_metrics(ngram_path, scenario_path, out_path, N):
    # Read Ngrams
    ngram_jsons = open(ngram_path, "r").readlines()
    entry_overlap_ngrams_list = []
    for ngram_json in ngram_jsons:
        entry_overlap_ngrams = json.loads(ngram_json)
        entry_overlap_ngrams_list.append(cattrs.structure(entry_overlap_ngrams, EntryOverlapNgrams))


    # create entry_overlap_ngrams_dict, a dict of DataOverlapStatsKey -> EntryOverlapNgrams
    entry_overlap_ngrams_dict = defaultdict(list)
    for entry_overlap_ngrams in entry_overlap_ngrams_list:
        entry_data_overlap_key = entry_overlap_ngrams.entry_data_overlap_key
        overlapping_ngram_counts = entry_overlap_ngrams.overlapping_ngram_counts
        ngram_count = entry_data_overlap_key.stats_key.overlap_protocol_spec.n
        stats_key = entry_data_overlap_key.stats_key
        if ngram_count not in [N]:
            continue
        entry_overlap_ngrams_dict[stats_key].append(entry_overlap_ngrams)

    # Read Scenarios
    light_scenarios = load_light_scenarios_from_jsonl(scenario_path)
    light_scenario_instance_dict = dict()
    for light_scenario in light_scenarios:
        instances = light_scenario.instances
        instance_dict = dict()
        for instance in instances:
            instance_dict[instance.id] = instance
        light_scenario_instance_dict[light_scenario.scenario_key] = instance_dict


    def compute_binary_overlap(instance_str, overlapping_ngram_counts, tokenizer, frequency = 0):
        """ 
        Compute  binary overlap
        If pass in frequency, include only the ngrams with count <= frequency
        """
        tokens = tokenizer.tokenize(instance_str)
        ngram_counts_dict = defaultdict(int)
        
        # construct a dict of ngram -> count
        for ngram, count in overlapping_ngram_counts:
            ngram = tuple(ast.literal_eval(ngram))
            ngram_counts_dict[ngram] = count

        metric_score = 0

        for ngram in ngrams(tokens, 13):
            count = ngram_counts_dict[ngram]
            if frequency == 0 or count <= frequency:
                if count != 0:
                    metric_score = 1
                    break

        overlap_metric = OverlapMetric(
            metric_score = metric_score,
            metric_protocol_spec = MetricProtocolSpec(
                partial_overlap_spec = PartialOverlapSpec.jaccard,
                frequency_spec = FrequencySpec(
                    filter_value = frequency,
                    weighting = False
                )
            )
        )

        return overlap_metric

    def compute_jaccard_overlap(instance_str, overlapping_ngram_counts, tokenizer, frequency = 0):
        """ 
        Compute weighted and unweighted jaccard overlap
        If pass in frequency, include only the ngrams with count <= frequency
        """
        tokens = tokenizer.tokenize(instance_str)
        ngram_counts_dict = defaultdict(int)
        
        # construct a dict of ngram -> count
        for ngram, count in overlapping_ngram_counts:
            ngram = tuple(ast.literal_eval(ngram))
            ngram_counts_dict[ngram] = count

        total_ngram_count = 0
        counts = 0
        weighted_score = 0

        for ngram in ngrams(tokens, 13):
            count = ngram_counts_dict[ngram]
            if frequency == 0 or count <= frequency:
                if count != 0:
                    counts += 1
                    weighted_score += 1 / count
            total_ngram_count += 1

        unweighted_score = counts / total_ngram_count
        weighted_score = weighted_score / total_ngram_count

        unweighted_overlap_metric = OverlapMetric(
            metric_score = unweighted_score ,
            metric_protocol_spec = MetricProtocolSpec(
                partial_overlap_spec = PartialOverlapSpec.jaccard,
                frequency_spec = FrequencySpec(
                    filter_value = frequency,
                    weighting = False
                )
            )
        )

        weighted_overlap_metric = OverlapMetric(
            metric_score = weighted_score ,
            metric_protocol_spec = MetricProtocolSpec(
                partial_overlap_spec = PartialOverlapSpec.jaccard,
                frequency_spec = FrequencySpec(
                    filter_value = frequency,
                    weighting = True
                )
            )
        )

        return unweighted_overlap_metric, weighted_overlap_metric

    # Token overlap
    def compute_token_overlap(instance_str, overlapping_ngram_counts, tokenizer, frequency = 0):
        """ 
        Compute weighted and unweighted token overlap
        If pass in frequency, include only the ngrams with count <= frequency
        """
        tokens = tokenizer.tokenize(instance_str)
        ngram_counts_dict = defaultdict(int)
        
        # construct a dict of ngram -> count
        for ngram, count in overlapping_ngram_counts:
            ngram = tuple(ast.literal_eval(ngram))
            ngram_counts_dict[ngram] = count

        total_token_count = 0
        counts = 0
        weighted_score = 0
        weight = 0
        token_budget = 0

        for ngram in ngrams(tokens, 13):
            curr_count = ngram_counts_dict[ngram]
            if frequency == 0 or count <= frequency:
                if curr_count != 0:
                    token_budget = 13
                    if weight > 0:
                        weight = min(curr_count, weight)
                    else:
                        weight = curr_count 

            if token_budget > 0:
                token_budget -= 1
                counts += 1
                weighted_score += 1 / weight
            else:
                weight = 0
            total_token_count += 1

        for token in ngram[1:]:
            if token_budget > 0:
                token_budget -= 1
                counts += 1
                weighted_score += 1 / weight
            total_token_count += 1

        unweighted_score = counts / total_token_count
        weighted_score = weighted_score / total_token_count

        unweighted_overlap_metric = OverlapMetric(
            metric_score = unweighted_score ,
            metric_protocol_spec = MetricProtocolSpec(
                partial_overlap_spec = PartialOverlapSpec.token,
                frequency_spec = FrequencySpec(
                    filter_value = frequency,
                    weighting = False
                )
            )
        )

        weighted_overlap_metric = OverlapMetric(
            metric_score = weighted_score ,
            metric_protocol_spec = MetricProtocolSpec(
                partial_overlap_spec = PartialOverlapSpec.token,
                frequency_spec = FrequencySpec(
                    filter_value = frequency,
                    weighting = True
                )
            )
        )

        return unweighted_overlap_metric, weighted_overlap_metric

    def compute_and_add_metrics(instance_str, overlapping_ngram_counts, tokenizer, entry_data_overlap_key, entry_overlap_metric_list, frequency = 0):

        overlap_metric = compute_binary_overlap(instance_str, overlapping_ngram_counts, tokenizer, frequency)
        binary_metric = EntryOverlapMetric(entry_data_overlap_key=entry_data_overlap_key, overlap_metric=overlap_metric)
        entry_overlap_metric_list.append(binary_metric)

        unweighted_overlap_metric, weighted_overlap_metric = compute_jaccard_overlap(instance_str, overlapping_ngram_counts, tokenizer, frequency)
        unweighted_jaccard = EntryOverlapMetric(entry_data_overlap_key=entry_data_overlap_key, overlap_metric=unweighted_overlap_metric)
        weighted_jaccard = EntryOverlapMetric(entry_data_overlap_key=entry_data_overlap_key, overlap_metric=weighted_overlap_metric)
        entry_overlap_metric_list.append(unweighted_jaccard)
        entry_overlap_metric_list.append(weighted_jaccard)

        unweighted_overlap_metric, weighted_overlap_metric = compute_token_overlap(instance_str, overlapping_ngram_counts, tokenizer, frequency)
        unweighted_token = EntryOverlapMetric(entry_data_overlap_key=entry_data_overlap_key, overlap_metric=unweighted_overlap_metric)
        weighted_token = EntryOverlapMetric(entry_data_overlap_key=entry_data_overlap_key, overlap_metric=weighted_overlap_metric)
        entry_overlap_metric_list.append(unweighted_token)
        entry_overlap_metric_list.append(weighted_token)

    def save_metrics_to_jsonl(overlap_metrics: List[EntryOverlapMetric], filename: str):
        with open(filename, "a") as f:
            for overlap_metric in overlap_metrics:
                f.write(json.dumps(asdict_without_nones(overlap_metric), ensure_ascii=False) + "\n")


    entry_overlap_metric_list = []
    tokenizer = get_tokenizer('default')
    for data_overlap_stats_key, entry_overlap_ngrams_list in entry_overlap_ngrams_dict.items():
        light_scenario_key = data_overlap_stats_key.light_scenario_key
        instance_dict = light_scenario_instance_dict[light_scenario_key]
        for entry_overlap_ngrams in entry_overlap_ngrams_list:
            entry_data_overlap_key = entry_overlap_ngrams.entry_data_overlap_key
            instance_id = entry_data_overlap_key.instance_id
            instance = instance_dict[instance_id]
            part = entry_data_overlap_key.part
            overlapping_ngram_counts = entry_overlap_ngrams.overlapping_ngram_counts
            if part == 'input':
                compute_and_add_metrics(instance.input, overlapping_ngram_counts, tokenizer, entry_data_overlap_key, entry_overlap_metric_list)
                compute_and_add_metrics(instance.input, overlapping_ngram_counts, tokenizer, entry_data_overlap_key, entry_overlap_metric_list, frequency=10)
            if part == 'references':
                reference = ' '.join(instance.references)
                compute_and_add_metrics(reference, overlapping_ngram_counts, tokenizer, entry_data_overlap_key, entry_overlap_metric_list)
                compute_and_add_metrics(reference, overlapping_ngram_counts, tokenizer, entry_data_overlap_key, entry_overlap_metric_list, frequency=10)

    save_metrics_to_jsonl(entry_overlap_metric_list, out_path)


ngram_path_base = 'output_stats_pi..gram_{}_ngrams'
scenario_path_base = './data/{}'
out_path_base = 'metrics_{}'

# Computing for N = 13 for illustrative purposes
N = 13

for char in range(ord('a'), ord('r') + 1):
    char_str = chr(char)
    ngram_path = ngram_path_base.format(f'xa{char_str}')
    scenario_path = scenario_path_base.format(f'xa{char_str}')
    out_path = out_path_base.format(f'xa{char_str}')
    print(ngram_path)
    
    # Call the get_metrics function with the constructed arguments
    get_metrics(ngram_path, scenario_path, out_path, N)

# ngram_path_base = 'output_stats_pi..gram_{}ngrams'
# scenario_path_base = './data/{}'
# out_path_base = 'metrics_{}'

# # Computing for N = 13 for illustrative purposes
# N = 13

# for char in range(ord('a'), ord('r') + 1):
#     char_str = chr(char)
#     ngram_path = ngram_path_base.format(f'xa{char_str}')
#     scenario_path = scenario_path_base.format(f'xa{char_str}')
#     out_path = out_path_base.format(f'xa{char_str}')
#     print(ngram_path)
    