from dataclasses import dataclass

from .perturbation import Perturbation
from .perturbation_description import PerturbationDescription


@dataclass
class LowerCasePerturbation(Perturbation):
    """
    Simpe perturbation turning input and references into lowercase.
    """

    @dataclass(frozen=True)
    class Description(PerturbationDescription):
        name: str

    name: str = "lowercase"

    @property
    def description(self) -> PerturbationDescription:
        return LowerCasePerturbation.Description(self.name)

    def perturb(self, text: str) -> str:
        return text.lower()
