from dataclasses import dataclass
from typing import Optional

from helm.benchmark.augmentations.perturbation_description import PerturbationDescription
from helm.benchmark.scenarios.scenario import Instance


@dataclass(frozen=True, eq=True)
class MetricName:
    # The name of the metric
    name: str

    # Split (e.g., train, valid, test)
    split: Optional[str] = None

    # Sub split (e.g., toxic, non-toxic)
    sub_split: Optional[str] = None

    # Description of the Perturbation applied to the Instances
    perturbation: Optional[PerturbationDescription] = None


@dataclass(frozen=True, eq=True)
class MetricContext:
    """Attributes determining groups of Instances we want to be aggregating over."""

    # Split (e.g., train, valid, test)
    split: Optional[str] = None

    # Sub split (e.g., toxic, non-toxic)
    sub_split: Optional[str] = None

    # Description of the Perturbation applied to the Instances
    perturbation: Optional[PerturbationDescription] = None

    @classmethod
    def from_instance(cls, instance: Instance):
        return cls(split=instance.split, sub_split=instance.sub_split, perturbation=instance.perturbation)

    @classmethod
    def from_metric_name(cls, metric_name: MetricName):
        return cls(split=metric_name.split, sub_split=metric_name.sub_split, perturbation=metric_name.perturbation)
