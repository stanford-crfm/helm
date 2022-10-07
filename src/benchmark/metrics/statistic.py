from dataclasses import dataclass, replace
import math
from typing import Dict, Optional

from .metric_name import MetricName


@dataclass
class Stat:
    """A mutable class that allows us to aggregate values and report mean/stddev."""

    name: MetricName
    count: int = 0
    sum: float = 0
    sum_squared: float = 0
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    variance: Optional[float] = None
    stddev: Optional[float] = None

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
        self._update_mean_variance_stddev()
        return self

    def merge(self, other: "Stat") -> "Stat":
        self.sum += other.sum
        self.sum_squared += other.sum_squared
        if other.count > 0:
            self.min = min(self.min, other.min) if self.count > 0 else other.min
        if other.max is not None:
            self.max = max(self.max, other.max) if self.count > 0 else other.max
        self.count += other.count
        self._update_mean_variance_stddev()
        return self

    def __repr__(self):
        return f"{self.name}[{self.bare_str()}]"

    def bare_str(self):
        if self.count > 0:
            return (
                f"min={self.min:.3f}, "
                f"mean={self.mean:.3f}, "
                f"max={self.max:.3f}, "
                f"sum={self.sum:.3f} "
                f"({self.count})"
            )
        else:
            return "(0)"

    def _update_mean(self):
        self.mean = self.sum / self.count if self.count else None

    def _update_variance(self):
        self._update_mean()
        if self.mean is None:
            return None
        pvariance = self.sum_squared / self.count - self.mean**2
        self.variance = 0 if pvariance < 0 else pvariance

    def _update_stddev(self):
        self._update_variance()
        self.stddev = math.sqrt(self.variance) if self.variance is not None else None

    def _update_mean_variance_stddev(self):
        self._update_stddev()

    def take_mean(self):
        """Return a version of the stat that only has the mean."""
        if self.count == 0:
            return self
        return Stat(self.name).add(self.mean)


def merge_stat(stats: Dict[MetricName, Stat], stat: Stat):
    """Mutate the appropriate part of `stats`."""
    if stat.name not in stats:
        # Important: copy so that we don't mutate accidentally
        stats[stat.name] = replace(stat)
    else:
        stats[stat.name].merge(stat)
