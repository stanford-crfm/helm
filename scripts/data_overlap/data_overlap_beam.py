"""Apache Beam specific code."""

import apache_beam as beam
import json

from typing import Dict, List, Iterable, Any, Set, DefaultDict, Tuple
from collections import defaultdict
from dataclasses import dataclass
from apache_beam.utils.shared import Shared
from data_overlap_spec import DataOverlapStats, DataOverlapStatsKey
from compute_data_overlap_metrics import compute_document_data_overlap
from common.general import asdict_without_nones


from compute_data_overlap_metrics import (
    load_light_scenarios_from_jsonl,
    create_ngram_index,
    NgramIndex,
)
from light_tokenizer import LightTokenizer


@dataclass(frozen=True)
class NgramIndexWrapper:
    """Wraps `NgramIndex` so that it can be shared."""

    ngram_index: NgramIndex


# Type alias
AllDataOverlapStats = Tuple[DefaultDict[DataOverlapStatsKey, Set], DefaultDict[DataOverlapStatsKey, Set]]


class ComputeDataOverlapStatsFn(beam.CombineFn):
    def __init__(
        self,
        scenario_data_path: str,
        n_values: List[int],
        normalization: str,
        shared_ngram_index: Shared,
        stats_key_counts: DefaultDict[DataOverlapStatsKey, int],
    ) -> None:
        self.scenario_data_path = scenario_data_path
        self.n_values = n_values
        self.normalization = normalization
        self.tokenizer: LightTokenizer = LightTokenizer()
        self.shared_ngram_index = shared_ngram_index
        self.stats_key_counts = stats_key_counts

    def setup(self, *args, **kwargs) -> None:
        self.scenarios = load_light_scenarios_from_jsonl(self.scenario_data_path)

        def init_shared_ngram_index():
            return NgramIndexWrapper(
                create_ngram_index(light_scenarios=self.scenarios, n_values=self.n_values, tokenizer=self.tokenizer, stats_key_counts=self.stats_key_counts)
            )

        self.ngram_index_wrapper = self.shared_ngram_index.acquire(init_shared_ngram_index)
        return super().setup(*args, **kwargs)

    def create_accumulator(self) -> AllDataOverlapStats:
        stats_key_to_input_ids: DefaultDict[DataOverlapStatsKey, Set] = defaultdict(set)
        stats_key_to_reference_ids: DefaultDict[DataOverlapStatsKey, Set] = defaultdict(set)
        return stats_key_to_input_ids, stats_key_to_reference_ids

    def add_input(
        self,
        stats_key_to_ids_tuple: AllDataOverlapStats,
        document: str,
    ) -> AllDataOverlapStats:
        stats_key_to_input_ids, stats_key_to_reference_ids = stats_key_to_ids_tuple

        # update all_overlap_stats in-place
        compute_document_data_overlap(
            document=document,
            ngram_index=self.ngram_index_wrapper.ngram_index,
            tokenizer=self.tokenizer,
            stats_key_to_input_ids=stats_key_to_input_ids,
            stats_key_to_reference_ids=stats_key_to_reference_ids,
        )
        return stats_key_to_input_ids, stats_key_to_reference_ids

    def merge_accumulators(self, accumulators: Iterable[AllDataOverlapStats]) -> AllDataOverlapStats:
        assert accumulators
        accumulators_iter = iter(accumulators)

        merged_stats_key_to_input_ids: DefaultDict[DataOverlapStatsKey, Set] = defaultdict(set)
        merged_stats_key_to_reference_ids: DefaultDict[DataOverlapStatsKey, Set] = defaultdict(set)

        for accumulator in accumulators_iter:
            stats_key_to_input_ids, stats_key_to_reference_ids = accumulator

            for key, value in stats_key_to_input_ids.items():
                merged_stats_key_to_input_ids[key].update(value)
            for key, value in stats_key_to_reference_ids.items():
                merged_stats_key_to_reference_ids[key].update(value)

        return merged_stats_key_to_input_ids, merged_stats_key_to_reference_ids

    def extract_output(self, accumulator: AllDataOverlapStats) -> List[DataOverlapStats]:
        print('\n\nHIIIII2\n')
        # print(accumulator)
        print(self.stats_key_counts)
        stats_key_to_input_ids, stats_key_to_reference_ids = accumulator
        all_data_overlap_stats = []
        for stats_key, count in self.stats_key_counts.items():
            data_overlap_stats = DataOverlapStats(
                data_overlap_stats_key=stats_key,
                instance_ids_with_overlapping_input=sorted(stats_key_to_input_ids[stats_key]),
                instance_ids_with_overlapping_reference=sorted(stats_key_to_reference_ids[stats_key]),
                num_instances=count,
            )
            all_data_overlap_stats.append(data_overlap_stats)
        return all_data_overlap_stats


def extract_summary_from_all_data_overlap_stats(
    all_data_overlap_stats: List[DataOverlapStats], tags: Dict[str, Any]
) -> str:
    # return "\n".join(json.dumps(overlap_stats.generate_summary(tags)) for overlap_stats in all_data_overlap_stats)
    print('\n\nHIIIII\n')
    print(all_data_overlap_stats)
    print("\n".join(
        json.dumps(asdict_without_nones(data_overlap_stats)) for data_overlap_stats in all_data_overlap_stats
    ))
    return "\n".join(
        json.dumps(asdict_without_nones(data_overlap_stats)) for data_overlap_stats in all_data_overlap_stats
    )


class ComputeAndWriteDataOverlapStats(beam.PTransform):
    def __init__(
        self, scenario_data_path: str, n_values: List[int], normalization: str, tags: Dict[str, Any], output_stats: str
    ):
        self.scenario_data_path = scenario_data_path
        self.n_values = n_values
        self.normalization = normalization
        self.tags = tags
        self.output_stats = output_stats
        self.stats_key_counts = defaultdict(int)

    def expand(self, pcollection: beam.PCollection):
        shared_ngram_index = Shared()
        return (
            pcollection
            | "ComputeOverlapStats"
            >> beam.CombineGlobally(
                ComputeDataOverlapStatsFn(
                    scenario_data_path=self.scenario_data_path,
                    n_values=self.n_values,
                    normalization=self.normalization,
                    shared_ngram_index=shared_ngram_index,
                    stats_key_counts=self.stats_key_counts,
                )
            )
            | "ExtractSummaryFromAllOverlapStats"
            # >> beam.io.WriteToText(self.output_stats)
            >> beam.Map(extract_summary_from_all_data_overlap_stats, tags=self.tags)
            | "WriteSummaries" >> beam.io.WriteToText(self.output_stats)
        )
