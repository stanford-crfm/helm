from dataclasses import dataclass

# Perturbation-relevant metrics, see computed_on below
PERTURBATION_ORIGINAL: str = "original"
PERTURBATION_PERTURBED: str = "perturbed"
PERTURBATION_WORST: str = "worst"


@dataclass(frozen=True)
class PerturbationDescription:
    """DataClass used to describe a Perturbation"""

    # Name of the Perturbation
    name: str

    # Whether a perturbation is relevant to robustness and/or fairness. Will be used to aggregate perturbations metrics
    robustness: bool = False
    fairness: bool = False

    # Which types of Instances we are evaluating, to be populated during metric evaluation. PERTURBATION_PERTURBED
    # (default) means we are evaluating on perturbed instances, PERTURBATION_ORIGINAL means we are evaluating the
    # unperturbed version of instances where this perturbation appplies, and, PERTURBATION_WORST means the the minimum
    # metric between the two.
    computed_on: str = PERTURBATION_PERTURBED
