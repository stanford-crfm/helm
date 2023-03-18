import json

from bitarray import bitarray
from typing import List, Optional

from helm.benchmark.contamination.light_scenario import LightScenario


PART_INPUT: str = "input"
PART_REF: str = "reference"


class ContaminationStats:
    """
    A memory-efficient class for contamination stats. The core data structures are bit arrays where
    every bit records whether an instance is dirty (contaminated) or not.
    """

    def __init__(self, scenario_spec: str, num_instances: int, stats_tags: Optional[List[str]] = None):
        self.scenario_spec = scenario_spec
        self.num_instances = num_instances
        self.stats_tags: List[str]
        if isinstance(stats_tags, list):
            self.stats_tags = stats_tags
        else:
            self.stats_tags = []

        self._input_bits = bitarray(num_instances)
        self._reference_bits = bitarray(num_instances)
        self._input_bits.setall(0)
        self._reference_bits.setall(0)

    @classmethod
    def from_scenario(cls, scenario: LightScenario, stats_tags: Optional[List[str]] = None):
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
        if self.scenario_spec != stats.scenario_spec:
            raise ValueError("Only stats for the same scenario can be merged.")
        if self.num_instances != stats.num_instances:
            raise ValueError("The sizes of the two scenarios need to equal.")
        self._input_bits |= stats._input_bits
        self._reference_bits |= stats._reference_bits

    @property
    def num_input_positive_instances(self):
        return self._input_bits.count()

    @property
    def num_reference_positive_instances(self):
        return self._reference_bits.count()

    @property
    def input_positive_rate(self):
        return self._input_bits.count() / self.num_instances

    @property
    def reference_positive_rate(self):
        return self._reference_bits.count() / self.num_instances

    @property
    def stats_repr(self) -> str:
        return f"{self.scenario_spec},{','.join(self.stats_tags)}"

    def generate_summary(self, tags: List[str]) -> str:
        """Output a summary of the stats"""
        summary = {
            "setting": f"{self.stats_repr}{'' if len(tags) == 0 else ','+','.join(tags)}",
            "total_instances": self.num_instances,
            "num_input_positive_instances": self.num_input_positive_instances,
            "num_reference_positive_instances": self.num_reference_positive_instances,
            "input_positive_rate": self.input_positive_rate,
            "reference_positive_rate": self.reference_positive_rate,
        }
        return json.dumps(summary)
