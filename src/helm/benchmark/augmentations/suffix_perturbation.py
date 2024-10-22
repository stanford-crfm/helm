from dataclasses import dataclass
from random import Random

from helm.benchmark.augmentations.perturbation import TextPerturbation
from helm.benchmark.augmentations.perturbation_description import PerturbationDescription


class SuffixPerturbation(TextPerturbation):
    """
    Appends a suffix to the end of the text. Example:

    A picture of a dog -> A picture of a dog, picasso
    """

    @dataclass(frozen=True)
    class Description(PerturbationDescription):
        suffix: str = ""

    name: str = "style"

    def __init__(self, suffix: str):
        self._suffix: str = suffix

    @property
    def description(self) -> PerturbationDescription:
        return SuffixPerturbation.Description(name=self.name, suffix=self._suffix)

    def perturb(self, text: str, rng: Random) -> str:
        return f"{text}, {self._suffix}"
