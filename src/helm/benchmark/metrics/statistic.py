from dataclasses import dataclass, replace
import math
from typing import Dict, Optional

from helm.benchmark.metrics.metric_name import MetricName


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
    """This is the population variance, not the sample variance.

    See https://towardsdatascience.com/variance-sample-vs-population-3ddbd29e498a
    for details.
    """

    stddev: Optional[float] = None
    """This is the population standard deviation, not the sample standard deviation.

    See https://towardsdatascience.com/variance-sample-vs-population-3ddbd29e498a
    for details.
    """

    def add(self, x) -> "Stat":
        # Skip Nones for statistic aggregation.
        if x is None:
            return self
        if isinstance(x, bool):
            x = 1 if x is True else 0
        self.sum += x
        self.sum_squared += x * x
        self.min = min(self.min, x) if self.min is not None else x
        self.max = max(self.max, x) if self.max is not None else x
        self.count += 1
        self._update_mean_variance_stddev()
        return self

    def merge(self, other: "Stat") -> "Stat":
        # Note: don't look at other.name
        self.sum += other.sum
        self.sum_squared += other.sum_squared
        if other.min is not None:
            self.min = min(self.min, other.min) if self.min is not None else other.min
        if other.max is not None:
            self.max = max(self.max, other.max) if self.max is not None else other.max
        self.count += other.count
        self._update_mean_variance_stddev()
        return self

    def __repr__(self):
        return f"{self.name}[{self.bare_str()}]"

    def bare_str(self) -> str:
        def process(x: Optional[float]) -> str:
            if x is None:
                return "None"
            if math.isnan(x):
                return "NaN"
            if int(x) == x:
                return str(int(x))
            return str(round(x, 3))

        if self.count > 0:
            return (
                f"min={process(self.min)}, "
                f"mean={process(self.mean)}, "
                f"max={process(self.max)}, "
                f"sum={process(self.sum)} "
                f"({self.count})"
            )
        else:
            return "(0)"

    def _update_mean_variance_stddev(self):
        if self.count == 0:
            # No stats with no elements.
            return
        # Update mean
        self.mean = self.sum / self.count
        # Update variance
        pvariance = self.sum_squared / self.count - self.mean**2
        self.variance = 0 if pvariance < 0 else pvariance
        # Update stddev
        self.stddev = math.sqrt(self.variance)

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
