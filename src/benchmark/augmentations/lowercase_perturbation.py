from dataclasses import dataclass
from typing import List

from .perturbation import Perturbation
from .perturbation_description import PerturbationDescription, ROBUSTNESS_TAG


class LowerCasePerturbation(Perturbation):
    """
    Simpe perturbation turning input and references into lowercase.
    """

    @dataclass(frozen=True)
    class Description(PerturbationDescription):
        name: str

    name: str = "lowercase"
    tags: List[str] = [ROBUSTNESS_TAG]

    @property
    def description(self) -> PerturbationDescription:
        return LowerCasePerturbation.Description(self.name, self.tags)

    def perturb(self, text: str) -> str:
        return text.lower()
