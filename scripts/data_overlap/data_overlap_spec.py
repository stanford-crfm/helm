from dataclasses import dataclass
from typing import List, Tuple

try:
    from light_scenario import LightScenarioKey
except Exception:
    from helm.benchmark.data_overlap.light_scenario import LightScenarioKey


@dataclass(frozen=True)
class GroupOverlapStats:
    """
    Dataclass that represents group data overlap stats
    e.g.
    {
        "group": "natural_qa_closedbook",
        "num_instances": 2144,
        "num_overlapping_inputs": 1,
        "num_overlapping_references": 100
    }
    """

    group: str

    num_instances: int

    num_overlapping_inputs: int

    num_overlapping_references: int

    @property
    def overlapping_input_ratio(self):
        return self.num_overlapping_inputs / self.num_instances

    @property
    def overlapping_reference_ratio(self):
        return self.num_overlapping_references / self.num_instances


@dataclass(frozen=True)
class OverlapProtocolSpec:
    """Specification for how we compute overlap"""

    # the N of the n_grams we're running
    n: int


@dataclass(frozen=True)
class DataOverlapStatsKey:
    """Dataclass that represents output data overlap stats"""

    light_scenario_key: LightScenarioKey

    overlap_protocol_spec: OverlapProtocolSpec


@dataclass(frozen=True)
class DataOverlapStats:
    """Dataclass that represents output data overlap stats"""

    data_overlap_stats_key: DataOverlapStatsKey

    num_instances: int

    instance_ids_with_overlapping_input: List[str]

    instance_ids_with_overlapping_reference: List[str]


@dataclass(frozen=True)
class EntryDataOverlapKey:
    """Unique key representing either the input or references of a single instance in a scenario."""

    stats_key: DataOverlapStatsKey
    part: str
    """Either PART_INPUT or PART_REF"""
    instance_id: str


@dataclass(frozen=True)
class EntryOverlapNgrams:
    """Dataclass that represents output data overlap stats"""

    entry_data_overlap_key: EntryDataOverlapKey

    overlapping_ngram_counts: List[Tuple[str, int]]
