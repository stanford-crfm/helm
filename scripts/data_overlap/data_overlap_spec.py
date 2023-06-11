from dataclasses import dataclass
from typing import List
from light_scenario import LightScenarioKey


@dataclass(frozen=True)
class OverlapProtocolSpec:
    """Specification for how we compute overlap"""

    # the N of the n_grams we're running
    N: int


@dataclass(frozen=True)
class OutputDataOverlapStatsKey:
    """Dataclass that represents output data overlap stats"""

    light_scenario_key: LightScenarioKey

    # the N of the n_grams we're running
    overlap_protocol_spec: OverlapProtocolSpec


@dataclass(frozen=True)
class OutputDataOverlapStats:
    """Dataclass that represents output data overlap stats"""

    output_data_overlap_stats_key: OutputDataOverlapStatsKey

    instance_ids_with_overlapping_input: List[str]

    instance_ids_with_overlapping_reference: List[str]
