"""Apache Beam specific code."""

from collections import defaultdict
from dataclasses import dataclass
import json
from typing import Dict, List, Iterable, Any, Set, DefaultDict, Tuple

import apache_beam as beam
from apache_beam.utils.shared import Shared

from data_overlap_spec import DataOverlapStats, DataOverlapStatsKey
from compute_data_overlap_metrics import compute_document_data_overlap
from common.general import asdict_without_nones
from common.util import get_tokenizer
from compute_data_overlap_metrics import (
    load_light_scenarios_from_jsonl,
    create_ngram_index,
    NgramIndex,
)


@dataclass(frozen=True)
class OverlapObjects:
    """
    Wraps `NgramIndex` and `stats_key_counts` so that it can be shared.
    https://beam.apache.org/releases/pydoc/2.48.0/apache_beam.utils.shared.html
    Several built-in types such as list and dict do not directly support weak references
    """

    ngram_index: NgramIndex
    stats_key_counts: DefaultDict[DataOverlapStatsKey, int]


# Type alias
AllDataOverlapStats = Tuple[DefaultDict[DataOverlapStatsKey, Set], DefaultDict[DataOverlapStatsKey, Set]]


class ComputeDataOverlapStatsFn(beam.CombineFn):
    def __init__(
        self,
        scenario_data_path: str,
        n_values: List[int],
        normalization: str,
        shared_overlap_objects: Shared,
    ) -> None:
        self.scenario_data_path = scenario_data_path
        self.n_values = n_values
        self.tokenizer = get_tokenizer(normalization)
        self.shared_overlap_objects = shared_overlap_objects

    def setup(self, *args, **kwargs) -> None:
        def init_shared_overlap_objects():
            scenarios = load_light_scenarios_from_jsonl(self.scenario_data_path)
            stats_key_counts: DefaultDict[DataOverlapStatsKey, int] = defaultdict(int)
            ngram_index = create_ngram_index(
                light_scenarios=scenarios,
                n_values=self.n_values,
                tokenizer=self.tokenizer,
                stats_key_counts=stats_key_counts,
            )
            return OverlapObjects(ngram_index=ngram_index, stats_key_counts=stats_key_counts)

        self.shared_overlap_objects = self.shared_overlap_objects.acquire(init_shared_overlap_objects)
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
        # update all_overlap_stats in-place
        compute_document_data_overlap(
            document=document,
            ngram_index=self.shared_overlap_objects.ngram_index,
            tokenizer=self.tokenizer,
            stats_key_to_input_ids=stats_key_to_ids_tuple[0],
            stats_key_to_reference_ids=stats_key_to_ids_tuple[1],
        )
        return stats_key_to_ids_tuple

    def merge_accumulators(self, accumulators: Iterable[AllDataOverlapStats]) -> AllDataOverlapStats:
        assert accumulators

        merged_stats_key_to_input_ids: DefaultDict[DataOverlapStatsKey, Set] = defaultdict(set)
        merged_stats_key_to_reference_ids: DefaultDict[DataOverlapStatsKey, Set] = defaultdict(set)

        for accumulator in accumulators:
            stats_key_to_input_ids, stats_key_to_reference_ids = accumulator

            for key, value in stats_key_to_input_ids.items():
                merged_stats_key_to_input_ids[key].update(value)
            for key, value in stats_key_to_reference_ids.items():
                merged_stats_key_to_reference_ids[key].update(value)

        return merged_stats_key_to_input_ids, merged_stats_key_to_reference_ids

    def extract_output(self, accumulator: AllDataOverlapStats) -> List[DataOverlapStats]:
        stats_key_to_input_ids, stats_key_to_reference_ids = accumulator
        all_data_overlap_stats = []
        for stats_key, count in self.shared_overlap_objects.stats_key_counts.items():
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

    def expand(self, pcollection: beam.PCollection):
        shared_overlap_objects = Shared()
        return (
            pcollection
            | "ComputeOverlapStats"
            >> beam.CombineGlobally(
                ComputeDataOverlapStatsFn(
                    scenario_data_path=self.scenario_data_path,
                    n_values=self.n_values,
                    normalization=self.normalization,
                    shared_overlap_objects=shared_overlap_objects,
                )
            )
            | "ExtractSummaryFromAllOverlapStats"
            >> beam.Map(extract_summary_from_all_data_overlap_stats, tags=self.tags)
            | "WriteSummaries" >> beam.io.WriteToText(self.output_stats)
        )
