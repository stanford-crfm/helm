from dataclasses import dataclass
from typing import List

from .perturbation import Perturbation
from .perturbation_description import PerturbationDescription, ROBUSTNESS_TAG


class ExtraSpacePerturbation(Perturbation):
    """
    A toy perturbation that replaces existing spaces in the text with
    `num_spaces` number of spaces.
    """

    @dataclass(frozen=True)
    class Description(PerturbationDescription):
        name: str
        num_spaces: int

    name: str = "extra_space"

    tags: List[str] = [ROBUSTNESS_TAG]

    def __init__(self, num_spaces: int):
        self.num_spaces = num_spaces

    @property
    def description(self) -> PerturbationDescription:
        return ExtraSpacePerturbation.Description(self.name, self.num_spaces)

    def perturb(self, text: str) -> str:
        return text.replace(" ", " " * self.num_spaces)
