from dataclasses import dataclass


@dataclass(frozen=True)
class PerturbationDescription:
    """DataClass used to describe a Perturbation"""

    # Name of the Perturbation
    name: str
    robustness: bool = False
    fairness: bool = False

    # Which types of Instances we are evaluating, to be populated during metric evaluation. "perturbed" (default) means
    # we are evaluating on perturbed instances, "original" means we are evaluating the unperturbed version of instances
    # where this perturbation appplies, and, "worst" means the the minimum metric between the two.
    computed_on: str = "perturbed"
