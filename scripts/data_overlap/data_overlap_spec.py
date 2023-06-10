from dataclasses import dataclass
from typing import List
from helm.benchmark.scenarios.scenario import ScenarioSpec


@dataclass(frozen=True)
class OverlapProtocolSpec:
    """Specification for how we compute overlap"""

    # the N of the n_grams we're running
    N: int


@dataclass(frozen=True)
class DataOverlapStats:
    """Dataclass that represents scenario level data overlap stats"""

    scenario_spec: ScenarioSpec

    # e.g. train vs test
    split: str

    # the N of the n_grams we're running
    overlap_protocol_spec: OverlapProtocolSpec

    num_instances: int

    instance_ids_with_overlapping_input: List[str]

    instance_ids_with_overlapping_reference: List[str]
