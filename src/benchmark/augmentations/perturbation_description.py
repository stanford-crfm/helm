from dataclasses import dataclass


@dataclass(frozen=True)
class PerturbationDescription:
    """DataClass used to describe a Perturbation"""

    # Name of the Perturbation
    name: str
    robustness: bool = False
    fairness: bool = False

    # Fields to capture which types of Instances we are evaluating, to be populated during metric evaluation.
    # `includes_perturbed` means we are evaluating on perturbed instances, `includes_identity` means we are evaluating
    # the unperturbed version of instances where this perturbation appplies, and, if both are true, we are
    # considering the minimum metric between the two.
    includes_perturbed: bool = True
    includes_identity: bool = False
