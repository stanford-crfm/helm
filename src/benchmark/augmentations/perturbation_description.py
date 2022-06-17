from dataclasses import dataclass
from typing import List


""" Perturbation tags """
ROBUSTNESS_TAG: str = "robustness"
FAIRNESS_TAG: str = "fairness"


@dataclass(frozen=True)
class PerturbationDescription:
    """DataClass used to describe a Perturbation"""

    # Name of the Perturbation
    name: str

    # Extra metadata (e.g., ["robustness"] or ["fairness"])
    # Used to group instances to compute metrics.
    tags: List[str]
