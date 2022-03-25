from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, List

from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import numpy as np

from benchmark.scenario import Instance
from .perturbation import Perturbation
from .perturbation_description import PerturbationDescription


# The implementation below is based on the following list of common misspellings:
# https://en.wikipedia.org/wiki/Wikipedia:Lists_of_common_misspellings/For_machines
@dataclass
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
        name: str

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
        self.detokenizer = TreebankWordDetokenizer()

    @property
    def description(self) -> PerturbationDescription:
        return MisspellingPerturbation.Description(self.name)

    def apply(self, instance: Instance, should_perturb_references: bool = True) -> Instance:
        assert instance.id is not None
        np.random.seed(int(instance.id[2:]))  # set seed based on instance ID
        return super().apply(instance, should_perturb_references)

    def perturb_word(self, word: str) -> str:
        # check if word is in dictionary and perturb with probability self.prob
        if word in self.correct_to_misspelling and np.random.rand() < self.prob:
            word = str(np.random.choice(self.correct_to_misspelling[word]))
        return word

    def perturb(self, text: str) -> str:
        words = word_tokenize(text)
        new_words = []
        for word in words:
            word = self.perturb_word(word)
            new_words.append(word)
        perturbed_str = str(self.detokenizer.detokenize(new_words))
        return perturbed_str
