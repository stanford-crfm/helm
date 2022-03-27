from typing import Dict

from benchmark.metric_name import MetricName
from common.statistic import Stat, merge_stat


def test_merge_stat():
    # Ensure that `MetricName`s are hashable
    metric_name = MetricName("some_metric", k=100)
    stats: Dict[MetricName, Stat] = {metric_name: Stat(metric_name).add(1)}
    merge_stat(stats, Stat(metric_name).add(1))

    assert len(stats) == 1
    assert stats[metric_name].sum == 2
