# mypy: check_untyped_defs = False
from typing import Dict
import re

from random import Random

from helm.common.general import match_case
from helm.benchmark.augmentations.perturbation import TextPerturbation
from helm.benchmark.augmentations.perturbation_description import PerturbationDescription


CONTRACTION_MAP: Dict[str, str] = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "could've": "could have",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I would",
    "I'll": "I will",
    "I'm": "I am",
    "I've": "I have",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "ma'am": "madam",
    "might've": "might have",
    "mightn't": "might not",
    "must've": "must have",
    "mustn't": "must not",
    "needn't": "need not",
    "oughtn't": "ought not",
    "shan't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "that'd": "that would",
    "that's": "that is",
    "there'd": "there would",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what're": "what are",
    "what's": "what is",
    "when's": "when is",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "won't": "will not",
    "would've": "would have",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have",
}


# The implementations below are based on
# https://github.com/GEM-benchmark/NL-Augmenter/blob/main/transformations/contraction_expansions/transformation.py
class ContractionPerturbation(TextPerturbation):
    """
    Contractions.
    Replaces each expansion with its contracted version.

    Perturbation example:

    **Input:**
        She is a doctor, and I am a student

    **Output:**
        She's a doctor, and I'm a student
    """

    name: str = "contraction"

    def __init__(self):
        self.contraction_map: Dict[str, str] = CONTRACTION_MAP
        self.reverse_contraction_map: Dict[str, str] = {value: key for key, value in self.contraction_map.items()}
        # Only contract things followed by a space to avoid contract end of sentence
        self.reverse_contraction_pattern = re.compile(
            r"\b({})\b ".format("|".join(self.reverse_contraction_map.keys())),
            flags=re.IGNORECASE | re.DOTALL,
        )

    @property
    def description(self) -> PerturbationDescription:
        return PerturbationDescription(name=self.name, robustness=True)

    def perturb(self, text: str, rng: Random) -> str:
        def cont(possible):
            match = possible.group(1)
            expanded_contraction = self.reverse_contraction_map.get(
                match, self.reverse_contraction_map.get(match.lower())
            )
            return match_case(match, expanded_contraction) + " "

        return self.reverse_contraction_pattern.sub(cont, text)


class ExpansionPerturbation(TextPerturbation):
    """
    Expansions.
    Replaces each contraction with its expanded version.

    Perturbation example:

    **Input:**
        She's a doctor, and I'm a student

    **Output:**
        She is a doctor, and I am a student
    """

    name: str = "expansion"

    def __init__(self):
        self.contraction_map: Dict[str, str] = CONTRACTION_MAP
        self.contraction_pattern = re.compile(
            r"\b({})\b".format("|".join(self.contraction_map.keys())),
            flags=re.IGNORECASE | re.DOTALL,
        )

    @property
    def description(self) -> PerturbationDescription:
        return PerturbationDescription(name=self.name, robustness=True)

    def perturb(self, text: str, rng: Random) -> str:
        def expand_match(contraction):
            match = contraction.group(0)
            expanded_contraction = self.contraction_map.get(match, self.contraction_map.get(match.lower()))
            return match_case(match, expanded_contraction)

        return self.contraction_pattern.sub(expand_match, text)
