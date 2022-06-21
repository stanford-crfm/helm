from dataclasses import dataclass


@dataclass(frozen=True)
class PerturbationDescription:
    """DataClass used to describe a Perturbation"""

    # Name of the Perturbation
    name: str
    robustness: bool = False
    fairness: bool = False
