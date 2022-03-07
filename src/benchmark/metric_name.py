from dataclasses import dataclass
from typing import Optional

from .augmentations.perturbation_description import PerturbationDescription


@dataclass(frozen=True, eq=True)
class MetricName:

    # The name of the metric
    name: str

    # For score_k
    k: Optional[int] = None

    # Split (e.g., train, valid, test)
    split: Optional[str] = None

    # Sub split (e.g. toxic, non-toxic)
    sub_split: Optional[str] = None

    # Description of the Perturbation applied to the Instances
    perturbation: Optional[PerturbationDescription] = None
