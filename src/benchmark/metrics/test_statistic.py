from typing import Dict, Set

from benchmark.augmentations.perturbation_description import PerturbationDescription
from .metric_name import MetricName
from .statistic import Stat, merge_stat


def test_merge_stat():
    # Ensure that `MetricName`s are hashable
    metric_name = MetricName("some_metric")
    stats: Dict[MetricName, Stat] = {metric_name: Stat(metric_name).add(1)}
    merge_stat(stats, Stat(metric_name).add(1))

    assert len(stats) == 1
    assert stats[metric_name].sum == 2


def test_hash():
    stats: Set[Stat] = set()
    stats.add(Stat(MetricName(name="metric", perturbation=PerturbationDescription(name="perturbation1"))))
    stats.add(Stat(MetricName(name="metric", perturbation=PerturbationDescription(name="perturbation1"))))
    assert len(stats) == 1

    # The perturbation is different
    stats.add(Stat(MetricName(name="metric", perturbation=PerturbationDescription(name="perturbation2"))))
    assert len(stats) == 2


def test_eq():
    stat1 = Stat(MetricName(name="metric", perturbation=PerturbationDescription(name="perturbation1")), count=1)
    stat2 = Stat(MetricName(name="metric", perturbation=PerturbationDescription(name="perturbation1")), count=3)
    assert stat1 == stat2, "Should be the same even with different counts"

    stat2 = Stat(MetricName(name="metric", perturbation=PerturbationDescription(name="perturbation2")), count=1)
    assert stat1 != stat2, "Not equal because of different perturbations"
