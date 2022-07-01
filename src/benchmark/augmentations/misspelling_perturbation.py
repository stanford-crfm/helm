from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Dict, List

import numpy as np

from benchmark.scenario import Instance
from common.general import match_case
from .perturbation import Perturbation
from .perturbation_description import PerturbationDescription


# The implementation below is based on the following list of common misspellings:
# https://en.wikipedia.org/wiki/Wikipedia:Lists_of_common_misspellings/For_machines
class MisspellingPerturbation(Perturbation):
    """
    Replaces words randomly with common misspellings, from a list of common misspellings.
    Perturbation example:
    Input:
        Already, the new product is not available.
    Output:
        Aready, the new product is not availible.
    """

    @dataclass(frozen=True)
    class Description(PerturbationDescription):
        prob: float = 0.0

    name: str = "misspellings"

    def __init__(self, prob: float):
        """
        prob (float): probability between [0,1] of perturbing a word to a
            common misspelling (if we have a common misspelling for the word)
        """
        self.prob: float = prob
        misspellings_file = Path(__file__).resolve().expanduser().parent / "correct_to_misspelling.json"
        with open(misspellings_file, "r") as f:
            self.correct_to_misspelling: Dict[str, List[str]] = json.load(f)

    @property
    def description(self) -> PerturbationDescription:
        return MisspellingPerturbation.Description(name=self.name, robustness=True, prob=self.prob)

    def apply(self, instance: Instance) -> Instance:
        assert instance.id is not None
        np.random.seed(int(instance.id[2:]))  # set seed based on instance ID
        return super().apply(instance)

    def perturb(self, text: str) -> str:
        mispelling_pattern = re.compile(
            r"\b({})\b".format("|".join(self.correct_to_misspelling.keys())), flags=re.IGNORECASE | re.DOTALL,
        )

        def mispell(match: re.Match) -> str:
            word = match.group(1)
            if np.random.rand() < self.prob:
                mispelled_word = str(np.random.choice(self.correct_to_misspelling[word.lower()]))
            return match_case(word, mispelled_word)

        return mispelling_pattern.sub(mispell, text)
