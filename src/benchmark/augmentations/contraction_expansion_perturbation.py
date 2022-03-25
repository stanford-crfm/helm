from dataclasses import dataclass
from typing import Dict
import re

from .perturbation import Perturbation
from .perturbation_description import PerturbationDescription


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
@dataclass
class ContractionPerturbation(Perturbation):
    """
    Contractions.
    Replaces each expansion with its contracted version.

    Perturbation example:
    Input:
        She is a doctor, and I am a student
    Output:
        She's a doctor, and I'm a student
    """

    @dataclass(frozen=True)
    class Description(PerturbationDescription):
        name: str

    name: str = "contraction"

    def __init__(self):
        self.contraction_map: Dict[str, str] = CONTRACTION_MAP
        self.reverse_contraction_map: Dict[str, str] = {value: key for key, value in self.contraction_map.items()}

    @property
    def description(self) -> PerturbationDescription:
        return ContractionPerturbation.Description(self.name)

    def contract(self, sentence: str) -> str:
        reverse_contraction_pattern = re.compile(
            r"\b({})\b ".format("|".join(self.reverse_contraction_map.keys())), flags=re.IGNORECASE | re.DOTALL,
        )

        def cont(possible):
            match = possible.group(1)
            # The first character is handled separately to preserve capitalization
            first_char = match[0]
            expanded_contraction = self.reverse_contraction_map.get(
                match, self.reverse_contraction_map.get(match.lower())
            )
            expanded_contraction = first_char + expanded_contraction[1:] + " "
            return expanded_contraction

        return reverse_contraction_pattern.sub(cont, sentence)

    def perturb(self, text: str) -> str:
        return self.contract(text)


@dataclass
class ExpansionPerturbation(Perturbation):
    """
    Expansions.
    Replaces each contraction with its expanded version.

    Perturbation example:
    Input:
        She's a doctor, and I'm a student
    Output:
        She is a doctor, and I am a student
    """

    @dataclass(frozen=True)
    class Description(PerturbationDescription):
        name: str

    name: str = "expansion"

    def __init__(self):
        self.contraction_map: Dict[str, str] = CONTRACTION_MAP

    @property
    def description(self) -> PerturbationDescription:
        return ExpansionPerturbation.Description(self.name)

    def expand_contractions(self, sentence):
        contraction_pattern = re.compile(
            r"\b({})\b".format("|".join(self.contraction_map.keys())), flags=re.IGNORECASE | re.DOTALL,
        )

        def expand_match(contraction):
            match = contraction.group(0)
            first_char = match[0]
            # The first character is handled separately to preserve capitalization
            expanded_contraction = self.contraction_map.get(match, self.contraction_map.get(match.lower()))
            expanded_contraction = first_char + expanded_contraction[1:]
            return expanded_contraction

        return contraction_pattern.sub(expand_match, sentence)

    def perturb(self, text: str) -> str:
        return self.expand_contractions(text)
