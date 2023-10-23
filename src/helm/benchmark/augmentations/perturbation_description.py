from dataclasses import dataclass
from typing import Optional

# Perturbation-relevant metrics, see computed_on below
PERTURBATION_ORIGINAL: str = "original"
PERTURBATION_PERTURBED: str = "perturbed"
PERTURBATION_WORST: str = "worst"


@dataclass(frozen=True)
class PerturbationDescription:
    """DataClass used to describe a Perturbation"""

    name: str
    """Name of the Perturbation"""

    robustness: bool = False
    """Whether a perturbation is relevant to robustness. Will be used to aggregate perturbations metrics"""

    fairness: bool = False
    """Whether a perturbation is relevant to fairness. Will be used to aggregate perturbations metrics"""

    computed_on: str = PERTURBATION_PERTURBED
    """Which types of Instances we are evaluating, to be populated during metric evaluation. PERTURBATION_PERTURBED
    (default) means we are evaluating on perturbed instances, PERTURBATION_ORIGINAL means we are evaluating the
    unperturbed version of instances where this perturbation applies, and, PERTURBATION_WORST means the the minimum
    metric between the two."""

    seed: Optional[int] = None
    """Seed added to instance_id when generating perturbation"""
