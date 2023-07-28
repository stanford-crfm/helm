from dataclasses import dataclass
from typing import List, Tuple
from light_scenario import LightScenarioKey


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
