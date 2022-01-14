from typing import Dict
from dataclasses import dataclass


@dataclass
class Stat:
    """A mutable class that allows us to aggregate values and report mean/stddev."""

    name: str
    count: int
    min: float
    max: float
    mean: float

    def __init__(self, name: str):
        self.name = name
        self.count = 0
        self.sum = 0
        self.sum_squared = 0
        self.min = float("+inf")
        self.max = float("-inf")

    def add(self, x) -> "Stat":
        if isinstance(x, bool):
            x = 1 if x is True else 0
        self.count += 1
        self.sum += x
        self.sum_squared += x * x
        self.min = min(self.min, x)
        self.max = max(self.max, x)
        return self

    def merge(self, other: "Stat") -> "Stat":
        self.count += other.count
        self.sum += other.sum
        self.sum_squared += other.sum_squared
        self.min = min(self.min, other.min)
        self.max = max(self.max, other.max)
        return self

    def __repr__(self):
        return f"{self.name}[min={self.min:.3f}, mean={self.mean:.3f}, max={self.max:.3f} ({self.count})]"

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
