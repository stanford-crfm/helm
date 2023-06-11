from dataclasses import dataclass
from typing import List
from light_scenario import LightScenarioKey


@dataclass(frozen=True)
class OverlapProtocolSpec:
    """Specification for how we compute overlap"""

    # the N of the n_grams we're running
    N: int

    def __eq__(self, other):
        return self.N == other.N


@dataclass(frozen=True)
class DataOverlapStatsKey:
    """Dataclass that represents output data overlap stats"""

    light_scenario_key: LightScenarioKey

    # the N of the n_grams we're running
    overlap_protocol_spec: OverlapProtocolSpec

    def __eq__(self, other):
        return (
            self.light_scenario_key == other.light_scenario_key
            and self.overlap_protocol_spec == other.overlap_protocol_spec
        )


@dataclass(frozen=True)
class DataOverlapStats:
    """Dataclass that represents output data overlap stats"""

    data_overlap_stats_key: DataOverlapStatsKey

    instance_ids_with_overlapping_input: List[str]

    instance_ids_with_overlapping_reference: List[str]

    def __eq__(self, other):
        return (
            self.data_overlap_stats_key == other.data_overlap_stats_key
            and self.instance_ids_with_overlapping_input == other.instance_ids_with_overlapping_input
            and self.instance_ids_with_overlapping_reference == other.instance_ids_with_overlapping_reference
        )
