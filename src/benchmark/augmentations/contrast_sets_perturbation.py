from dataclasses import dataclass, replace

import random
from typing import Sequence

from benchmark.scenario import Instance, Reference
from .perturbation_description import PerturbationDescription
from .perturbation import Perturbation


@dataclass
class ContrastSetsPerturbation(Perturbation):
    """
    Contrast Sets are from the paper and are supported for the BoolQ and the IMDB scenario:
        https://arxiv.org/abs/2004.02709

    Original repository can be found at:
        https://github.com/allenai/contrast-sets

    An example instance of a perturbation for the BoolQ dataset (from the original paper):

    The Fate of the Furious premiered in Berlin on April 4, 2017, and was theatrically released in the
    United States on April 14, 2017, playing in 3D, IMAX 3D and 4DX internationally. . . A spinoff film starring
    Johnson and Statham’s characters is scheduled for release in August 2019, while the ninth and tenth films are
    scheduled for releases on the years 2020 and 2021.
    question: is “Fate and the Furious” the last movie?
    answer: no

    perturbed question: is “Fate and the Furious” the first of multiple movies?
    perturbed answer: yes
    perturbation strategy: adjective change.

    An example instance of a perturbation for the IMDB dataset(from the original paper):

    Orginal instance: Hardly one to be faulted for his ambition or his vision,
    it is genuinely unexpected, then, to see all Park’s effort add up to so very little. . . .
    The premise is promising, gags are copious and offbeat humour
    abounds but it all fails miserably to create any meaningful connection with the audience.
    Sentiment: negative

    Perturbed instance: Hardly one to be faulted for his ambition or his vision, here we see all Park’s effort come to
    fruition. . . . The premise is perfect, gags are hilarious and offbeat humour abounds, and it
    creates a deep connection with the audience.
    Sentiment: positive
    """

    @dataclass(frozen=True)
    class Description(PerturbationDescription):
        name: str

    name: str = "contrast_sets"

    def __init__(self):
        pass

    @property
    def description(self) -> PerturbationDescription:
        return ContrastSetsPerturbation.Description(self.name)

    def apply(self, instance: Instance, should_perturb_references: bool = True) -> Instance:
        """
        Generates a new Instance by perturbing the input, tagging the Instance and perturbing the References,
        if should_perturb_references is true.
        """

        assert should_perturb_references
        random.seed(0)

        perturbed_instance: str = instance.input
        perturbed_references: Sequence[Reference] = instance.references

        if instance.contrast_inputs is not None and instance.contrast_references is not None:
            perturb_index: int = random.choice(range(len(instance.contrast_inputs)))
            perturbed_instance = instance.contrast_inputs[perturb_index]
            perturbed_references = instance.contrast_references[perturb_index]

        return replace(
            instance, input=perturbed_instance, references=perturbed_references, perturbation=self.description,
        )

    def perturb(self, text: str) -> str:
        """How to perturb the text. """
        pass
