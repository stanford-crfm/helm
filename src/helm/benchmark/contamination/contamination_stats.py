from bitarray import bitarray
from typing import Optional, Dict, Any

from helm.benchmark.contamination.light_scenario import LightScenario


PART_INPUT: str = "input"
PART_REF: str = "reference"


class ContaminationStats:
    """
    A memory-efficient class for contamination stats. The core data structures are bit arrays where
    every bit records whether an instance is dirty (contaminated) or not.
    """

    def __init__(self, scenario_spec: Dict[str, Any], num_instances: int, stats_tags: Optional[Dict[str, Any]] = None):
        self.stats_spec = {"scenario_spec": scenario_spec}
        self.num_instances = num_instances
        if stats_tags is not None:
            self.stats_spec.update(stats_tags)

        self._input_bits = bitarray(num_instances)
        self._reference_bits = bitarray(num_instances)
        self._input_bits.setall(0)
        self._reference_bits.setall(0)

    @classmethod
    def from_scenario(cls, scenario: LightScenario, stats_tags: Optional[Dict[str, Any]] = None):
        return cls(
            scenario_spec=scenario.scenario_spec, num_instances=len(scenario.light_instances), stats_tags=stats_tags
        )

    def write_dirty(self, instance_id: int, part: str):
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
        if self.stats_spec != stats.stats_spec:
            raise ValueError("Only stats with the same `stats_spec` can be merged.")
        if self.num_instances != stats.num_instances:
            raise ValueError("The sizes of the two scenarios need to equal.")
        self._input_bits |= stats._input_bits
        self._reference_bits |= stats._reference_bits

    @property
    def num_instances_with_dirty_input(self):
        return self._input_bits.count()

    @property
    def num_instances_with_dirty_reference(self):
        return self._reference_bits.count()

    @property
    def dirty_input_fraction(self):
        return self._input_bits.count() / self.num_instances

    @property
    def dirty_reference_fraction(self):
        return self._reference_bits.count() / self.num_instances

    def generate_summary(self, summary_tags: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Output a summary of the stats"""
        if summary_tags is None:
            summary_tags = {}
        summary = {
            "setting": {**self.stats_spec, **summary_tags},
            "num_instances": self.num_instances,
            "num_instances_with_dirty_input": self.num_instances_with_dirty_input,
            "num_instances_with_dirty_reference": self.num_instances_with_dirty_reference,
            "dirty_input_fraction": self.dirty_input_fraction,
            "dirty_reference_fraction": self.dirty_reference_fraction,
        }
        return summary
