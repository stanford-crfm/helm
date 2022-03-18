from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import List
import numpy as np
import json
from pathlib import Path
from ntlk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

from .perturbation_description import PerturbationDescription
from benchmark.scenario import Instance, Reference
from common.object_spec import ObjectSpec, create_object


class Perturbation(ABC):

    # Unique name to describe perturbation
    name: str

    @property
    def description(self) -> PerturbationDescription:
        """Description of the perturbation."""
        return PerturbationDescription(self.name)

    def apply(self, instance_id: str, instance: Instance, should_perturb_references: bool = True) -> Instance:
        """
        Generates a new Instance by perturbing the input, tagging the Instance and perturbing the References,
        if should_perturb_references is true.
        """
        references: List[Reference] = instance.references
        if should_perturb_references:
            references = [self.perturb_reference(reference) for reference in references]

        return replace(
            instance,
            input=self.perturb(instance.input),
            references=references,
            id=instance_id,
            perturbation=self.description,
        )

    def perturb_reference(self, reference: Reference) -> Reference:
        """Generates a new Reference by perturbing the output and tagging the Reference."""
        return replace(reference, output=self.perturb(reference.output), tags=reference.tags + [self.name])

    @abstractmethod
    def perturb(self, text: str) -> str:
        """How to perturb the text. """
        pass


class PerturbationSpec(ObjectSpec):
    """Defines how to instantiate Perturbation."""

    pass


def create_perturbation(perturbation_spec: PerturbationSpec) -> Perturbation:
    """Creates Perturbation from PerturbationSpec."""
    return create_object(perturbation_spec)


@dataclass
class IdentityPerturbation(Perturbation):
    """Doesn't apply any perturbations."""

    name: str = "identity"

    def perturb(self, text: str) -> str:
        return text


@dataclass
class ExtraSpacePerturbation(Perturbation):
    """
    A toy perturbation that replaces existing spaces in the text with
    `num_spaces` number of spaces.
    """

    @dataclass(frozen=True)
    class Description(PerturbationDescription):
        name: str
        num_spaces: int

    name: str = "extra_space"

    def __init__(self, num_spaces: int):
        self.num_spaces = num_spaces

    @property
    def description(self) -> PerturbationDescription:
        return ExtraSpacePerturbation.Description(self.name, self.num_spaces)

    def perturb(self, text: str) -> str:
        return text.replace(" ", " " * self.num_spaces)


# The implementation below is based on the following list of common misspellings:
# https://en.wikipedia.org/wiki/Wikipedia:Lists_of_common_misspellings/For_machines
@dataclass
class Misspellings(Perturbation):
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
		'''
		prob (float): probability between [0,1] of perturbing a word to a common misspelling (if we have a common misspelling for the word)
		'''
        self.prob = prob
		misspellings_file = Path(__file__).resolve().expanduser().parent / 'correct_to_misspelling.json'
		with open(misspellings_file, 'r') as f:
			self.correct_to_misspelling = json.load(f)
        self.detokenizer = TreebankWordDetokenizer()

    @property
    def description(self) -> PerturbationDescription:
        return Misspellings.Description(self.name)

    def count_upper(self, word: str) -> int:
        return sum(c.isupper() for c in word)

    def perturb_word(self, word: str) -> str:
        perturbed = False
        if word in self.correct_to_misspelling and np.random.rand() < self.prob:
            word = np.random.choice(self.correct_to_misspelling[word])
            perturbed = True
        return word, perturbed

    def perturb_seed(self, text: str, seed: int=0) -> str:
        words = word_tokenize(text)
        new_words = []
        for word in words:
            # check if word in original form is in dictionary
            word, perturbed = self.perturb_word(word)
            if not perturbed and self.count_upper(word) == 1 and word[0].isupper():
                # if only the first letter is upper case
                # check if the lowercase version is in dict
                lowercase_word = word.lower()
                word, perturbed = self.perturb_word(lowercase_word)
                word[0] = word[0].upper()
            new_words.append(word)
        return self.detokenizer.detokenize(new_words)

    def perturb(self, text: str) -> str:
        return self.perturb_seed(text)
