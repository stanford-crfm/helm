from dataclasses import dataclass
from random import Random

from helm.benchmark.augmentations.perturbation_description import PerturbationDescription
from helm.benchmark.augmentations.perturbation import TextPerturbation


class TyposPerturbation(TextPerturbation):
    """
    Typos. For implementation details, see
    https://github.com/GEM-benchmark/NL-Augmenter/tree/main/transformations/butter_fingers_perturbation

    Replaces each random letters with nearby keys on a querty keyboard.
    We modified the keyboard mapping compared to the NL-augmenter augmentations so that: a) only distance-1 keys are
    used for replacement, b) the original letter is no longer an option, c) removed special characters (e.g., commas).

    Perturbation example:

    **Input:**
        After their marriage, she started a close collaboration with Karvelas.

    **Output:**
        Aftrr theif marriage, she started a close collaboration with Karcelas.
    """

    @dataclass(frozen=True)
    class Description(PerturbationDescription):
        prob: float = 0.0

    name: str = "typos"

    def __init__(self, prob: float):
        self.prob: float = prob

    @property
    def description(self) -> PerturbationDescription:
        return TyposPerturbation.Description(name=self.name, robustness=True, prob=self.prob)

    def perturb(self, text: str, rng: Random) -> str:
        key_approx = {}

        key_approx["q"] = "was"
        key_approx["w"] = "qesad"
        key_approx["e"] = "wsdfr"
        key_approx["r"] = "edfgt"
        key_approx["t"] = "rfghy"
        key_approx["y"] = "tghju"
        key_approx["u"] = "yhjki"
        key_approx["i"] = "ujklo"
        key_approx["o"] = "iklp"
        key_approx["p"] = "ol"

        key_approx["a"] = "qwsz"
        key_approx["s"] = "weadzx"
        key_approx["d"] = "erfcxs"
        key_approx["f"] = "rtgvcd"
        key_approx["g"] = "tyhbvf"
        key_approx["h"] = "yujnbg"
        key_approx["j"] = "uikmnh"
        key_approx["k"] = "iolmj"
        key_approx["l"] = "opk"

        key_approx["z"] = "asx"
        key_approx["x"] = "sdcz"
        key_approx["c"] = "dfvx"
        key_approx["v"] = "fgbc"
        key_approx["b"] = "ghnv"
        key_approx["n"] = "hjmb"
        key_approx["m"] = "jkn"

        perturbed_texts = ""
        for letter in text:
            lcletter = letter.lower()
            if lcletter not in key_approx.keys():
                new_letter = lcletter
            else:
                if rng.random() < self.prob:
                    new_letter = rng.choice(list(key_approx[lcletter]))
                else:
                    new_letter = lcletter
            # go back to original case
            if not lcletter == letter:
                new_letter = new_letter.upper()
            perturbed_texts += new_letter
        return perturbed_texts
