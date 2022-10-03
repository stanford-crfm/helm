from dataclasses import replace

from random import Random
from typing import Sequence, Optional
import numpy as np

from benchmark.scenarios.scenario import Instance, Reference
from .perturbation_description import PerturbationDescription
from .perturbation import Perturbation


class OptionPermutationsPerturbation(Perturbation):
    """
    Permute the set of references. Intended to measure the robustness of the model to prompt ordering.
    """

    name: str = "option_permutations"

    should_perturb_references: bool = True

    def __init__(self):
        pass

    @property
    def description(self) -> PerturbationDescription:
        return PerturbationDescription(name=self.name, robustness=False)

    def apply(self, instance: Instance, seed: Optional[int] = None) -> Instance:
        """
        Generates a new Instance by perturbing the input, tagging the Instance and
        perturbing the References, if `should_perturb_references` is true.
        """
        rng: Random = self.get_rng(instance, seed)

        perturbed_instance: str = instance.input

        outputs = [pr.output for pr in instance.references]
        if outputs[-1].lower() in ["none", "refused", "none of the above", "all of the above"]:
            last, rest = outputs[-1], outputs[:-1]
            rest = rest[::-1]
            rng.shuffle(rest)
            outputs = rest + [last]
        else:
            rng.shuffle(outputs)

        perturbed_references = []
        for i in range(len(instance.references)):
            ref = Reference(output=outputs[i], tags=instance.references[i].tags)
            perturbed_references.append(ref)

        description = replace(self.description, seed=seed)
        return replace(instance, input=perturbed_instance, references=perturbed_references, perturbation=description,)

    def perturb(self, text: str, rng: Random) -> str:  # we need this since parent method is abstract
        pass
