from typing import Dict

import pytest
import statistics

from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.statistic import Stat, merge_stat


def test_stat_add():
    stat = Stat(MetricName("some_metric"))
    population = list(range(10))
    for i in population:
        stat.add(i)
    assert stat.sum == sum(population)
    assert stat.count == 10
    assert stat.min == 0
    assert stat.max == 9
    assert stat.mean == sum(population) / 10
    assert stat.variance == pytest.approx(statistics.pvariance(population))
    assert stat.stddev == pytest.approx(statistics.pstdev(population))


def test_merge_stat():
    # Ensure that `MetricName`s are hashable
    metric_name = MetricName("some_metric")
    stats: Dict[MetricName, Stat] = {metric_name: Stat(metric_name).add(1)}
    merge_stat(stats, Stat(metric_name).add(1))

    assert len(stats) == 1
    assert stats[metric_name].sum == 2
    assert stats[metric_name].mean == 1


def test_merge_empty_stat():
    # This test ensures we guard against division by zero.
    metric_name = MetricName("some_metric")
    empty_1 = Stat(metric_name)
    empty_2 = Stat(metric_name)
    merged = empty_1.merge(empty_2)

    assert merged.count == 0
    assert merged.stddev is None
