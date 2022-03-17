from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import List
import random, itertools

import ipdb

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


# The implementation below is based on
# https://github.com/GEM-benchmark/NL-Augmenter/tree/main/transformations/butter_fingers_perturbation
@dataclass
class Typos(Perturbation):
    """
    Typos. For implementation details, see
    https://github.com/GEM-benchmark/NL-Augmenter/tree/main/transformations/butter_fingers_perturbation
    Replaces each expansion with its typo.
    Perturbation example:
    Input:
        After their marriage, she started a close collaboration with Karvelas.
    Output:
        After theif marriage, she started a close collaboration with Karxelas.
    """

    @dataclass(frozen=True)
    class Description(PerturbationDescription):
        name: str

    name: str = "typos"

    def __init__(self, prob):
        self.prob = prob

    @property
    def description(self) -> PerturbationDescription:
        return Typos.Description(self.name)

    def butter_finger(self, text, keyboard="querty", seed=0, max_outputs=1):
        random.seed(seed)
        key_approx = {}

        if keyboard == "querty":
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
        else:
            print("Keyboard not supported.")

        prob_of_typo = int(self.prob * 100)
        perturbed_texts = ""
        for letter in text:
            lcletter = letter.lower()
            if lcletter not in key_approx.keys():
                new_letter = lcletter
            else:
                if random.choice(range(0, 100)) <= prob_of_typo:
                    new_letter = random.choice(key_approx[lcletter])
                else:
                    new_letter = lcletter
            # go back to original case
            if not lcletter == letter:
                new_letter = new_letter.upper()
            perturbed_texts += new_letter
        return perturbed_texts

    def perturb(self, text: str) -> str:
        return self.butter_finger(text)
