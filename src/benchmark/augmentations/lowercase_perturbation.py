from dataclasses import dataclass

from .perturbation import Perturbation
from .perturbation_description import PerturbationDescription


class LowerCasePerturbation(Perturbation):
    """
    Simpe perturbation turning input and references into lowercase.
    """

    @dataclass(frozen=True)
    class Description(PerturbationDescription):
        name: str
        robustness: bool
        fairness: bool

    name: str = "lowercase"

    @property
    def description(self) -> PerturbationDescription:
        return LowerCasePerturbation.Description(name=self.name, robustness=True, fairness=False)

    def perturb(self, text: str) -> str:
        return text.lower()
