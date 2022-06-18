from dataclasses import dataclass
import random
from typing import List

from benchmark.scenario import Instance
from .perturbation import Perturbation
from .perturbation_description import PerturbationDescription, ROBUSTNESS_TAG


class SpacePerturbation(Perturbation):
    """
    A simple perturbation that replaces existing spaces with 0-max_spaces spaces (thus potentially merging words).
    """

    @dataclass(frozen=True)
    class Description(PerturbationDescription):
        name: str
        tags: List[str]
        max_spaces: int

    name: str = "space"
    tags: List[str] = [ROBUSTNESS_TAG]

    def __init__(self, max_spaces: int):
        self.max_spaces = max_spaces

    @property
    def description(self) -> PerturbationDescription:
        return SpacePerturbation.Description(self.name, self.tags, self.max_spaces)

    def apply(self, instance: Instance, should_perturb_references: bool = True) -> Instance:
        assert instance.id is not None
        random.seed(int(instance.id[2:]))  # set seed based on instance ID
        return super().apply(instance, should_perturb_references)

    def perturb(self, text: str) -> str:
        result = []
        for word in text.split(" "):
            result.append(word)
            result.append(" " * random.randint(0, self.max_spaces))
        return "".join(result[:-1])
