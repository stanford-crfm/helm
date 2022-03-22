from dataclasses import dataclass

from .perturbation import Perturbation


@dataclass
class IdentityPerturbation(Perturbation):
    """Doesn't apply any perturbations."""

    name: str = "identity"

    def perturb(self, text: str) -> str:
        return text
