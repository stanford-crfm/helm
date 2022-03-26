from dataclasses import dataclass

import numpy as np

from benchmark.scenario import Instance
from .perturbation_description import PerturbationDescription
from .perturbation import Perturbation


@dataclass
class TyposPerturbation(Perturbation):
    """
    Typos. For implementation details, see
    https://github.com/GEM-benchmark/NL-Augmenter/tree/main/transformations/butter_fingers_perturbation
    Replaces each random letters with nearby keys on a querty keyboard.

    Perturbation example:
    Input:
        After their marriage, she started a close collaboration with Karvelas.
    Output:
        Aftrr theif marriage, she started a close collaboration with Karcelas.
    """

    @dataclass(frozen=True)
    class Description(PerturbationDescription):
        name: str
        prob: float

    name: str = "TyposPerturbation"

    def __init__(self, prob: float):
        self.prob: float = prob

    @property
    def description(self) -> PerturbationDescription:
        return TyposPerturbation.Description(self.name, self.prob)

    def butter_finger(self, text):
        key_approx = {}

        key_approx["q"] = "qwasedzx"
        key_approx["w"] = "wqesadrfcx"
        key_approx["e"] = "ewrsfdqazxcvgt"
        key_approx["r"] = "retdgfwsxcvgt"
        key_approx["t"] = "tryfhgedcvbnju"
        key_approx["y"] = "ytugjhrfvbnji"
        key_approx["u"] = "uyihkjtgbnmlo"
        key_approx["i"] = "iuojlkyhnmlp"
        key_approx["o"] = "oipklujm"
        key_approx["p"] = "plo['ik"

        key_approx["a"] = "aqszwxwdce"
        key_approx["s"] = "swxadrfv"
        key_approx["d"] = "decsfaqgbv"
        key_approx["f"] = "fdgrvwsxyhn"
        key_approx["g"] = "gtbfhedcyjn"
        key_approx["h"] = "hyngjfrvkim"
        key_approx["j"] = "jhknugtblom"
        key_approx["k"] = "kjlinyhn"
        key_approx["l"] = "lokmpujn"

        key_approx["z"] = "zaxsvde"
        key_approx["x"] = "xzcsdbvfrewq"
        key_approx["c"] = "cxvdfzswergb"
        key_approx["v"] = "vcfbgxdertyn"
        key_approx["b"] = "bvnghcftyun"
        key_approx["n"] = "nbmhjvgtuik"
        key_approx["m"] = "mnkjloik"
        key_approx[" "] = " "

        perturbed_texts = ""
        for letter in text:
            lcletter = letter.lower()
            if lcletter not in key_approx.keys():
                new_letter = lcletter
            else:
                if np.random.rand() < self.prob:
                    new_letter = np.random.choice(list(key_approx[lcletter]))
                else:
                    new_letter = lcletter
            # go back to original case
            if not lcletter == letter:
                new_letter = new_letter.upper()
            perturbed_texts += new_letter
        return perturbed_texts

    def apply(self, instance: Instance, should_perturb_references: bool = True) -> Instance:
        assert instance.id is not None
        np.random.seed(int(instance.id[2:]))  # set seed based on instance ID
        return super().apply(instance, should_perturb_references)

    def perturb(self, text: str) -> str:
        return self.butter_finger(text)
