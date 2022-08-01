from dataclasses import dataclass
import random

from benchmark.scenarios.scenario import Instance
from .perturbation import Perturbation
from .perturbation_description import PerturbationDescription


class SpacePerturbation(Perturbation):
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

    def apply(self, instance: Instance) -> Instance:
        assert instance.id is not None
        random.seed(int(instance.id[2:]))  # set seed based on instance ID
        return super().apply(instance)

    def perturb(self, text: str) -> str:
        result = []
        for word in text.split(" "):
            result.append(word)
            result.append(" " * random.randint(0, self.max_spaces))
        return "".join(result[:-1])
