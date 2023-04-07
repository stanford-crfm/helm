"""Apache Beam specific code."""

import apache_beam as beam
import json

from typing import Dict, List, Iterable, Any
from dataclasses import dataclass
from apache_beam.utils.shared import Shared


from helm.benchmark.contamination.compute_contamination_metrics import (
    load_light_scenarios_from_jsonl,
    create_ngram_index,
    create_all_contamination_stats,
    compute_scenario_document_contamination,
    get_light_tokenizer,
    AllContaminationStats,
    NgramIndex,
)
from helm.benchmark.contamination.light_tokenizer import LightTokenizer


@dataclass(frozen=True)
class NgramIndexWrapper:
    """Wraps `NgramIndex` so that it can be shared."""

    ngram_index: NgramIndex


class ComputeContaminationStatsFn(beam.CombineFn):
    def __init__(
        self,
        scenario_data_path: str,
        n_values: List[int],
        normalization: str,
        shared_ngram_index: Shared,
    ) -> None:
        self.scenario_data_path = scenario_data_path
        self.n_values = n_values
        self.tokenizer: LightTokenizer = get_light_tokenizer(normalization)
        self.shared_ngram_index = shared_ngram_index

    def setup(self, *args, **kwargs) -> None:
        self.scenarios = load_light_scenarios_from_jsonl(self.scenario_data_path)

        def init_shared_ngram_index():
            return NgramIndexWrapper(
                create_ngram_index(light_scenarios=self.scenarios, n_values=self.n_values, tokenizer=self.tokenizer)
            )

        self.ngram_index_wrapper = self.shared_ngram_index.acquire(init_shared_ngram_index)
        return super().setup(*args, **kwargs)

    def create_accumulator(self) -> AllContaminationStats:
        return create_all_contamination_stats(light_scenarios=self.scenarios, n_values=self.n_values)

    def add_input(self, all_contamination_stats: AllContaminationStats, document: str) -> AllContaminationStats:
        # update all_contamination_stats in-place
        compute_scenario_document_contamination(
            document=document,
            ngram_index=self.ngram_index_wrapper.ngram_index,
            all_contamination_stats=all_contamination_stats,
            tokenizer=self.tokenizer,
        )
        return all_contamination_stats

    def merge_accumulators(self, accumulators: Iterable[AllContaminationStats]) -> AllContaminationStats:
        assert accumulators
        merged_accumulator = accumulators[0]
        for accumulator in accumulators[1:]:
            for contamination_stats_key, contamination_stats in accumulator:
                merged_accumulator[contamination_stats_key].merge(contamination_stats)
        return merged_accumulator

    def extract_output(self, accumulator: AllContaminationStats) -> AllContaminationStats:
        return accumulator


def extract_summary_from_all_contamination_stats(
    all_contamination_stats: AllContaminationStats, tags: Dict[str, Any]
) -> str:
    return "\n".join(
        json.dumps(contamination_stats.generate_summary(tags))
        for contamination_stats in all_contamination_stats.values()
    )


class ComputeAndWriteContaminationStats(beam.PTransform):
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
            | "ComputeContaminationStats"
            >> beam.CombineGlobally(
                ComputeContaminationStatsFn(
                    scenario_data_path=self.scenario_data_path,
                    n_values=self.n_values,
                    normalization=self.normalization,
                    shared_ngram_index=shared_ngram_index,
                )
            )
            | "ExtractSummaryFromAllContaminationStats"
            >> beam.Map(extract_summary_from_all_contamination_stats, tags=self.tags)
            | "WriteSummaries" >> beam.io.WriteToText(self.output_stats)
        )
