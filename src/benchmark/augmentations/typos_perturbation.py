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

        key_approx["q"] = "qwas"
        key_approx["w"] = "wqesad"
        key_approx["e"] = "ewsdfr"
        key_approx["r"] = "redfgt"
        key_approx["t"] = "trfghy"
        key_approx["y"] = "ytghju"
        key_approx["u"] = "uyhjki"
        key_approx["i"] = "iujklo"
        key_approx["o"] = "oikl;p"
        key_approx["p"] = "pol;'["

        key_approx["a"] = "aqwsz"
        key_approx["s"] = "sweadzx"
        key_approx["d"] = "derfcxs"
        key_approx["f"] = "frtgvcd"
        key_approx["g"] = "gtyhbvf"
        key_approx["h"] = "hyujnbg"
        key_approx["j"] = "juikmnh"
        key_approx["k"] = "kiol,mj"
        key_approx["l"] = "lop;.,k"

        key_approx["z"] = "zasx"
        key_approx["x"] = "xsdcz"
        key_approx["c"] = "cdfvx"
        key_approx["v"] = "vfgbc"
        key_approx["b"] = "bghnv"
        key_approx["n"] = "nhjmb"
        key_approx["m"] = "mjk,n"
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
