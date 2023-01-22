from dataclasses import dataclass
from random import Random

from helm.proxy.clients.google_translate_client import GoogleTranslateClient
from .perturbation import Perturbation
from .perturbation_description import PerturbationDescription


class TranslatePerturbation(Perturbation):
    """
    Translates to different languages.
    """

    @dataclass(frozen=True)
    class Description(PerturbationDescription):
        language_code: str = "zh-CN"

    name: str = "translate"

    def __init__(self, language_code: str):
        self.language_code: str = language_code
        self.google_translate_client = GoogleTranslateClient()

    @property
    def description(self) -> PerturbationDescription:
        return TranslatePerturbation.Description(name=self.name, language_code=self.language_code)

    def perturb(self, text: str, rng: Random) -> str:
        return self.google_translate_client.translate(text, self.language_code)
