from dataclasses import replace

from random import Random
from typing import List, Optional

from helm.benchmark.scenarios.scenario import Instance, Reference, Input
from helm.benchmark.augmentations.perturbation_description import PerturbationDescription
from helm.benchmark.augmentations.perturbation import Perturbation


class ContrastSetsPerturbation(Perturbation):
    """
    Contrast Sets are from this paper (currently supported for the BoolQ and IMDB scenarios):
    https://arxiv.org/abs/2004.02709

    Original repository can be found at:
    https://github.com/allenai/contrast-sets

    An example instance of a perturbation for the BoolQ dataset:

    ```
    The Fate of the Furious premiered in Berlin on April 4, 2017, and was theatrically released in the United States on
    April 14, 2017, playing in 3D, IMAX 3D and 4DX internationally. . . A spinoff film starring Johnson and Statham’s
    characters is scheduled for release in August 2019, while the ninth and tenth films are scheduled for releases on
    the years 2020 and 2021.
    question: is “Fate and the Furious” the last movie?
    answer: no

    perturbed question: is “Fate and the Furious” the first of multiple movies?
    perturbed answer: yes
    perturbation strategy: adjective change.
    ```

    An example instance of a perturbation for the IMDB dataset (from the original paper):

    ```
    Original instance: Hardly one to be faulted for his ambition or his vision, it is genuinely unexpected, then, to see
    all Park’s effort add up to so very little. . . .  The premise is promising, gags are copious and offbeat humour
    abounds but it all fails miserably to create any meaningful connection with the audience.
    Sentiment: negative

    Perturbed instance: Hardly one to be faulted for his ambition or his vision, here we
    see all Park’s effort come to fruition. . . . The premise is perfect, gags are
    hilarious and offbeat humour abounds, and it creates a deep connection with the
    audience.
    Sentiment: positive
    ```
    """

    name: str = "contrast_sets"

    # Contrast sets do not make sense if not True.
    should_perturb_references: bool = True

    def __init__(self):
        pass

    @property
    def description(self) -> PerturbationDescription:
        return PerturbationDescription(name=self.name, robustness=True)

    def apply(self, instance: Instance, seed: Optional[int] = None) -> Instance:
        """
        Generates a new Instance by perturbing the input, tagging the Instance and
        perturbing the References, if `should_perturb_references` is true.
        """
        rng: Random = self.get_rng(instance, seed)

        perturbed_input: Input = instance.input
        perturbed_references: List[Reference] = instance.references

        if instance.contrast_inputs is not None and instance.contrast_references is not None:
            perturb_index: int = rng.choice(range(len(instance.contrast_inputs)))
            perturbed_input = instance.contrast_inputs[perturb_index]
            perturbed_references = instance.contrast_references[perturb_index]

        description: PerturbationDescription = replace(self.description, seed=seed)
        return replace(
            instance,
            input=perturbed_input,
            references=perturbed_references,
            perturbation=description,
        )
