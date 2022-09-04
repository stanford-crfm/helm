from random import Random
import re

from .perturbation_description import PerturbationDescription
from .perturbation import Perturbation
from .lowercase_perturbation import LowerCasePerturbation
from .contraction_expansion_perturbation import ContractionPerturbation
from .space_perturbation import SpacePerturbation
from .misspelling_perturbation import MisspellingPerturbation


class RobustnessPerturbation(Perturbation):
    """
    Canonical robustness perturbation that composes several perturbations.
    These perturbations are chosen to be reasonable.
    """

    name: str = "robustness"

    should_perturb_references: bool = True

    def __init__(self):
        self.lowercase_perturbation = LowerCasePerturbation()
        self.contraction_perturbation = ContractionPerturbation()
        self.space_perturbation = SpacePerturbation(max_spaces=3)
        self.misspelling_perturbation = MisspellingPerturbation(prob=0.1)

    @property
    def description(self) -> PerturbationDescription:
        return PerturbationDescription(name=self.name, robustness=True)

    def perturb(self, text: str, rng: Random) -> str:
        ### Preserve word identity

        # Lowercase the words
        text = self.lowercase_perturbation.perturb(text, rng)

        ### Preserve well-formedness

        # Contractions
        text = self.contraction_perturbation.perturb(text, rng)

        # Change quote types
        text = re.sub(r'"', "'", text)

        ### Perturb words (might make things a bit ill-formed)

        # Misspellings
        text = self.misspelling_perturbation.perturb(text, rng)

        # Replace hyphenated words ("pro-democracy" => "pro democracy" or "prodemocracy")
        text = re.sub(r"-", lambda x: rng.choice([" ", ""]), text)

        ### Change spaces around words

        # Insert extra spaces between words
        text = self.space_perturbation.perturb(text, rng)

        return text
