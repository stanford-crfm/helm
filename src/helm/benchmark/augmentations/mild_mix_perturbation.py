from random import Random

from helm.benchmark.augmentations.perturbation_description import PerturbationDescription
from helm.benchmark.augmentations.perturbation import TextPerturbation
from helm.benchmark.augmentations.lowercase_perturbation import LowerCasePerturbation
from helm.benchmark.augmentations.contraction_expansion_perturbation import ContractionPerturbation
from helm.benchmark.augmentations.space_perturbation import SpacePerturbation
from helm.benchmark.augmentations.misspelling_perturbation import MisspellingPerturbation


class MildMixPerturbation(TextPerturbation):
    """
    Canonical robustness perturbation that composes several perturbations.
    These perturbations are chosen to be reasonable.
    """

    name: str = "mild_mix"

    # Don't perturb references because it's not fair to have to generate broken text.
    should_perturb_references: bool = False

    def __init__(self):
        self.lowercase_perturbation = LowerCasePerturbation()
        self.contraction_perturbation = ContractionPerturbation()
        self.space_perturbation = SpacePerturbation(max_spaces=3)
        self.misspelling_perturbation = MisspellingPerturbation(prob=0.1)

    @property
    def description(self) -> PerturbationDescription:
        return PerturbationDescription(name=self.name, robustness=True)

    def perturb(self, text: str, rng: Random) -> str:
        # 1) Preserve word identity

        # Lowercase the words
        text = self.lowercase_perturbation.perturb(text, rng)

        # 2) Preserve well-formedness

        # Contractions
        text = self.contraction_perturbation.perturb(text, rng)

        # 3) Perturb words (might make things a bit ill-formed)

        # Misspellings
        text = self.misspelling_perturbation.perturb(text, rng)

        # 4) Change spaces around words

        # Insert extra spaces between words
        text = self.space_perturbation.perturb(text, rng)

        return text
