from dataclasses import dataclass


""" Perturbation tags """
ROBUSTNESS_TAG: str = "robustness"
FAIRNESS_TAG: str = "fairness"


@dataclass(frozen=True)
class PerturbationDescription:
    """DataClass used to describe a Perturbation"""

    # Name of the Perturbation
    name: str
