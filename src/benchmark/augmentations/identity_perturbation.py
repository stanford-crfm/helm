from typing import List

from .perturbation import Perturbation
from .perturbation_description import ROBUSTNESS_TAG


class IdentityPerturbation(Perturbation):
    """Doesn't apply any perturbations."""

    name: str = "identity"
    tags: List[str] = [ROBUSTNESS_TAG]

    def perturb(self, text: str) -> str:
        return text
