"""Apache Beam specific code."""

import apache_beam as beam
import json

from typing import Dict, List, Iterable, Any
from dataclasses import dataclass
from apache_beam.utils.shared import Shared


from compute_data_overlap_metrics import (
    load_light_scenarios_from_jsonl,
    create_ngram_index,
    create_all_data_overlap_stats,
    compute_scenario_document_data_overlap,
    AllDataOverlapStats,
    NgramIndex,
)
from light_tokenizer import LightTokenizer


@dataclass(frozen=True)
class NgramIndexWrapper:
    """Wraps `NgramIndex` so that it can be shared."""

    ngram_index: NgramIndex


class ComputeDataOverlapStatsFn(beam.CombineFn):
    def __init__(
        self,
        scenario_data_path: str,
        n_values: List[int],
        normalization: str,
        shared_ngram_index: Shared,
    ) -> None:
        self.scenario_data_path = scenario_data_path
        self.n_values = n_values
        self.tokenizer: LightTokenizer = LightTokenizer()
        self.shared_ngram_index = shared_ngram_index

    def setup(self, *args, **kwargs) -> None:
        self.scenarios = load_light_scenarios_from_jsonl(self.scenario_data_path)

        def init_shared_ngram_index():
            return NgramIndexWrapper(
                create_ngram_index(light_scenarios=self.scenarios, n_values=self.n_values, tokenizer=self.tokenizer)
            )

        self.ngram_index_wrapper = self.shared_ngram_index.acquire(init_shared_ngram_index)
        return super().setup(*args, **kwargs)

    def create_accumulator(self) -> AllDataOverlapStats:
        return create_all_data_overlap_stats(light_scenarios=self.scenarios, n_values=self.n_values)

    def add_input(self, all_overlap_stats: AllDataOverlapStats, document: str) -> AllDataOverlapStats:
        # update all_overlap_stats in-place
        compute_scenario_document_data_overlap(
            document=document,
            ngram_index=self.ngram_index_wrapper.ngram_index,
            all_overlap_stats=all_overlap_stats,
            tokenizer=self.tokenizer,
        )
        return all_overlap_stats

    def merge_accumulators(self, accumulators: Iterable[AllDataOverlapStats]) -> AllDataOverlapStats:
        assert accumulators
        accumulators_iter = iter(accumulators)
        merged_accumulator = next(accumulators_iter)
        for accumulator in accumulators_iter:
            for overlap_stats_key, overlap_stats in accumulator.items():
                merged_accumulator[overlap_stats_key].merge(overlap_stats)
        return merged_accumulator

    def extract_output(self, accumulator: AllDataOverlapStats) -> AllDataOverlapStats:
        return accumulator


def extract_summary_from_all_data_overlap_stats(all_overlap_stats: AllDataOverlapStats, tags: Dict[str, Any]) -> str:
    return "\n".join(json.dumps(overlap_stats.generate_summary(tags)) for overlap_stats in all_overlap_stats.values())


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
                )
            )
            | "ExtractSummaryFromAllOverlapStats"
            >> beam.Map(extract_summary_from_all_data_overlap_stats, tags=self.tags)
            | "WriteSummaries" >> beam.io.WriteToText(self.output_stats)
        )
