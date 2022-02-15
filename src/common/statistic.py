from typing import Dict, List
from dataclasses import dataclass, field
import random


@dataclass
class Stat:
    """A mutable class that allows us to aggregate values and report mean/stddev."""

    name: str
    count: int
    min: float
    max: float
    sum: float
    mean: float
    values: List[float] = field(default_factory=list)

    def __init__(self, name: str, values_buffer_size: int = 300):
        self.name = name
        self.values_buffer_size = values_buffer_size
        self.count = 0
        self.sum = 0
        self.sum_squared = 0
        self.min = float("+inf")
        self.max = float("-inf")
        self.values = []

    def _add_to_values(self, x: float):
        if len(self.values) < self.values_buffer_size:
            self.values.append(x)
        else:
            # Remove existing value from sketch with probability n/n+1.
            index_to_remove = random.randint(0, self.values_buffer_size)
            if index_to_remove < self.values_buffer_size:
                self.values[index_to_remove] = x

    def add(self, x) -> "Stat":
        if isinstance(x, bool):
            x = 1 if x is True else 0
        self.count += 1
        self.sum += x
        self.sum_squared += x * x
        self.min = min(self.min, x)
        self.max = max(self.max, x)
        self._add_to_values(x)
        return self

    def merge(self, other: "Stat") -> "Stat":
        self.count += other.count
        self.sum += other.sum
        self.sum_squared += other.sum_squared
        self.min = min(self.min, other.min)
        self.max = max(self.max, other.max)
        for x in other.values:
            self._add_to_values(x)
        return self

    def __repr__(self):
        return (
            f"{self.name}[min={self.min:.3f}, mean={self.mean:.3f}, max={self.max:.3f}, "
            f"sum={self.sum:.3f} ({self.count})]"
        )

    @property
    def mean(self):
        if self.count == 0:
            return float("nan")
        return self.sum / self.count

    @property
    def stddev(self):
        if self.count == 0:
            return float("nan")
        return self.sum_squared / self.count - self.mean ** 2

    def take_mean(self):
        """Return a version of the stat that only has the mean."""
        return Stat(self.name).add(self.mean)


def merge_stat(stats: Dict[str, Stat], stat: Stat):
    """Mutate the appropriate part of `stats`."""
    if stat.name not in stats:
        stats[stat.name] = stat
    else:
        stats[stat.name].merge(stat)
