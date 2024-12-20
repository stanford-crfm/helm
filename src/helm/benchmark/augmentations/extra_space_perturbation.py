from dataclasses import dataclass
from random import Random

from helm.benchmark.augmentations.perturbation import TextPerturbation
from helm.benchmark.augmentations.perturbation_description import PerturbationDescription


class ExtraSpacePerturbation(TextPerturbation):
    """
    A toy perturbation that replaces existing spaces in the text with
    `num_spaces` number of spaces.
    """

    @dataclass(frozen=True)
    class Description(PerturbationDescription):
        num_spaces: int = 0

    name: str = "extra_space"

    def __init__(self, num_spaces: int):
        self.num_spaces = num_spaces

    @property
    def description(self) -> PerturbationDescription:
        return ExtraSpacePerturbation.Description(name=self.name, robustness=True, num_spaces=self.num_spaces)

    def perturb(self, text: str, rng: Random) -> str:
        return text.replace(" ", " " * self.num_spaces)
