from bitarray import bitarray
from typing import Optional, Dict, Any, Hashable
from dataclasses import dataclass

from helm.benchmark.contamination.light_scenario import LightScenario, LightScenarioKey
from helm.common.general import asdict_without_nones


PART_INPUT: str = "input"
PART_REF: str = "reference"


@dataclass(frozen=True)
class ContaminationStatsKey:
    """Unique key representing a `ContaminationStats` instance."""

    metadata: Dict[str, Hashable]

    def __hash__(self):
        return hash(tuple((k, self.metadata[k]) for k in sorted(self.metadata.keys())))


class ContaminationStats:
    """
    A memory-efficient class for contamination stats. The core data structures are bit arrays where
    every bit records whether an instance is contaminated or not.
    """

    def __init__(
        self, light_scenario_key: LightScenarioKey, num_instances: int, stats_tags: Optional[Dict[str, Any]] = None
    ):
        self.stats_key = ContaminationStatsKey(metadata={"light_scenario_key": light_scenario_key})
        self.num_instances = num_instances
        if stats_tags is not None:
            self.stats_key.metadata.update(stats_tags)

        self._input_bits = bitarray(num_instances)
        self._reference_bits = bitarray(num_instances)
        self._input_bits.setall(0)
        self._reference_bits.setall(0)

    @classmethod
    def from_scenario(cls, scenario: LightScenario, stats_tags: Optional[Dict[str, Any]] = None):
        return cls(
            light_scenario_key=scenario.light_scenario_key,
            num_instances=len(scenario.light_instances),
            stats_tags=stats_tags,
        )

    def write_one_to_bit(self, instance_id: int, part: str):
        if part == PART_INPUT:
            self._input_bits[instance_id] = 1
        elif part == PART_REF:
            self._reference_bits[instance_id] = 1
        else:
            raise ValueError(f"There is no valid part of instance named {part}")

    def get_bit(self, instance_id: int, part: str) -> int:
        if part == PART_INPUT:
            return self._input_bits[instance_id]
        elif part == PART_REF:
            return self._reference_bits[instance_id]
        else:
            raise ValueError(f"There is no valid part of instance named {part}")

    def merge(self, stats):
        """Merge two stats instance of the same scenario"""
        if self.stats_key != stats.stats_key:
            raise ValueError("Only stats with the same `stats_key` can be merged.")
        if self.num_instances != stats.num_instances:
            raise ValueError("The sizes of the two scenarios need to equal.")
        self._input_bits |= stats._input_bits
        self._reference_bits |= stats._reference_bits

    @property
    def num_instances_with_contaminated_input(self):
        return self._input_bits.count()

    @property
    def num_instances_with_contaminated_reference(self):
        return self._reference_bits.count()

    @property
    def contaminated_input_fraction(self):
        return self._input_bits.count() / self.num_instances

    @property
    def contaminated_reference_fraction(self):
        return self._reference_bits.count() / self.num_instances

    def generate_summary(self, summary_tags: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Output a summary of the stats"""
        if summary_tags is None:
            summary_tags = {}
        summary = {
            "setting": {"stats_key": asdict_without_nones(self.stats_key), **summary_tags},
            "num_instances": self.num_instances,
            "num_instances_with_contaminated_input": self.num_instances_with_contaminated_input,
            "num_instances_with_contaminated_reference": self.num_instances_with_contaminated_reference,
            "contaminated_input_fraction": self.contaminated_input_fraction,
            "contaminated_reference_fraction": self.contaminated_reference_fraction,
        }
        return summary
