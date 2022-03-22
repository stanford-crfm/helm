from abc import ABC, abstractmethod
<<<<<<< HEAD
from dataclasses import dataclass, replace
from typing import List, Dict
import re
=======
from dataclasses import replace
from typing import List
>>>>>>> master

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

    def apply(self, instance: Instance, should_perturb_references: bool = True) -> Instance:
        """
        Generates a new Instance by perturbing the input, tagging the Instance and perturbing the References,
        if should_perturb_references is true.
        """
        references: List[Reference] = instance.references
        if should_perturb_references:
            references = [self.perturb_reference(reference) for reference in references]

        # Don't modify `id` of `Instance` here.
        # All the perturbed Instances generated from a single Instance should have the same ID.
        return replace(
            instance, input=self.perturb(instance.input), references=references, perturbation=self.description,
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
<<<<<<< HEAD


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


# The implementation below is based on
# https://github.com/GEM-benchmark/NL-Augmenter/blob/main/transformations/contraction_expansions/transformation.py
@dataclass
class Contraction(Perturbation):
    """
    Contractions. For implementation details, see
    https://github.com/GEM-benchmark/NL-Augmenter/tree/main/transformations/contraction_expansions

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
        self.contraction_map: Dict[str] = {
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
        self.reverse_contraction_map: Dict[str] = {value: key for key, value in self.contraction_map.items()}

    @property
    def description(self) -> PerturbationDescription:
        return Contraction.Description(self.name)

    def contract(self, sentence: str) -> str:
        """
        The implementation here is based on
        https://github.com/GEM-benchmark/NL-Augmenter/tree/main/transformations/contraction_expansions"
        """
        reverse_contraction_pattern = re.compile(
            r"\b({})\b ".format("|".join(self.reverse_contraction_map.keys())), flags=re.IGNORECASE | re.DOTALL,
        )

        def cont(possible):
            match = possible.group(1)
            first_char = match[0]
            expanded_contraction = self.reverse_contraction_map.get(
                match, self.reverse_contraction_map.get(match.lower())
            )
            expanded_contraction = first_char + expanded_contraction[1:] + " "
            return expanded_contraction

        return reverse_contraction_pattern.sub(cont, sentence)

    def perturb(self, text: str) -> str:
        return self.contract(text)
=======
>>>>>>> master
