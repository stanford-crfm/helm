from dataclasses import dataclass
from typing import Optional

from .augmentations.perturbation_description import PerturbationDescription
from .scenarios.scenario import Instance


@dataclass(frozen=True, eq=True)
class MetricName:

    # The name of the metric
    name: str

    # K of Top K score
    # Whenever the metric depends on completions, k needs to be an int that captures how many completions were used to
    # compute it (e.g., exact_match, f1_score). This will be k=1 in most cases.
    # If a metric is independent of completions, then k should be None to reflect that (e.g., co2 cost, perplexity).
    k: Optional[int] = None

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
