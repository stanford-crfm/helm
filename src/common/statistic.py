from typing import Dict, List
from dataclasses import dataclass, field
import random

from benchmark.metric_name import MetricName


@dataclass
class Stat:
    """A mutable class that allows us to aggregate values and report mean/stddev."""

    name: MetricName
    count: int
    min: float
    max: float
    sum: float
    mean: float
    values: List[float] = field(default_factory=list)

    def __init__(self, name: MetricName, values_buffer_size: int = 300):
        self.name = name
        self.values_buffer_size = values_buffer_size
        self.count = 0
        self.sum = 0
        self.sum_squared = 0
        # We don't set these to +infinity/-infinity because those don't serialize to JSON well.
        self.min = None
        self.max = None
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
        # Skip Nones for statistic aggregation.
        if x is None:
            return self
        if isinstance(x, bool):
            x = 1 if x is True else 0
        self.sum += x
        self.sum_squared += x * x
        self.min = min(self.min, x) if self.count > 0 else x
        self.max = max(self.max, x) if self.count > 0 else x
        self.count += 1
        self._add_to_values(x)
        return self

    def merge(self, other: "Stat") -> "Stat":
        self.sum += other.sum
        self.sum_squared += other.sum_squared
        if other.count > 0:
            self.min = min(self.min, other.min) if self.count > 0 else other.min
        if other.max is not None:
            self.max = max(self.max, other.max) if self.count > 0 else other.max
        self.count += other.count
        for x in other.values:
            self._add_to_values(x)
        return self

    def __repr__(self):
        if self.count > 0:
            return (
                f"{self.name}["
                f"min={self.min:.3f}, "
                f"mean={self.mean:.3f}, "
                f"max={self.max:.3f}, "
                f"sum={self.sum:.3f} "
                f"({self.count})]"
            )
        else:
            return f"{self.name}[(0)]"

    @property
    def mean(self):
        if self.count == 0:
            return None
        return self.sum / self.count

    @property
    def stddev(self):
        if self.count == 0:
            return None
        return self.sum_squared / self.count - self.mean ** 2

    def take_mean(self):
        """Return a version of the stat that only has the mean."""
        if self.count == 0:
            return self
        return Stat(self.name).add(self.mean)

def merge_stat(stats: Dict[MetricName, Stat], stat: Stat):
    """Mutate the appropriate part of `stats`."""
    if stat.name not in stats:
        stats[stat.name] = stat
    else:
        stats[stat.name].merge(stat)
