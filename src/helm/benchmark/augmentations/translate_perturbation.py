from dataclasses import dataclass
from random import Random

from helm.clients.google_translate_client import GoogleTranslateClient
from helm.benchmark.augmentations.perturbation import TextPerturbation
from helm.benchmark.augmentations.perturbation_description import PerturbationDescription


class TranslatePerturbation(TextPerturbation):
    """
    Translates to different languages.
    """

    @dataclass(frozen=True)
    class Description(PerturbationDescription):
        # Language code to translate to. Needs a default value since we inherit from `PerturbationDescription`
        language_code: str = "zh-CN"

    name: str = "translate"
    should_perturb_references: bool = True

    def __init__(self, language_code: str):
        self.language_code: str = language_code
        self.google_translate_client = GoogleTranslateClient()

    @property
    def description(self) -> PerturbationDescription:
        return TranslatePerturbation.Description(name=self.name, language_code=self.language_code)

    def perturb(self, text: str, rng: Random) -> str:
        return self.google_translate_client.translate(text, self.language_code)
