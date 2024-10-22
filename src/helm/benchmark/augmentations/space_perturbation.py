from dataclasses import dataclass
from random import Random
import re

from helm.benchmark.augmentations.perturbation import TextPerturbation
from helm.benchmark.augmentations.perturbation_description import PerturbationDescription


class SpacePerturbation(TextPerturbation):
    """
    A simple perturbation that replaces existing spaces with 0-max_spaces spaces (thus potentially merging words).
    """

    @dataclass(frozen=True)
    class Description(PerturbationDescription):
        max_spaces: int = 0

    name: str = "space"

    def __init__(self, max_spaces: int):
        self.max_spaces = max_spaces

    @property
    def description(self) -> PerturbationDescription:
        return SpacePerturbation.Description(name=self.name, robustness=True, max_spaces=self.max_spaces)

    def perturb(self, text: str, rng: Random) -> str:
        # Replace each space with a random number of spaces
        return re.sub(r" +", lambda x: " " * rng.randint(1, self.max_spaces), text)
