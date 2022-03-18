import itertools
import os
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import List, Set

import spacy

from benchmark.scenario import Instance, Reference
from common.object_spec import ObjectSpec, create_object
from .perturbation_description import PerturbationDescription


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


def pairwise(iterable):
    # "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    # From https://docs.python.org/3.8/library/itertools.html#itertools.pairwise
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


@dataclass
class CityNameReplacementPerturbation(Perturbation):
    """
    A perturbation that replaces the name of a populous city to less populated
    ones. This implements B20 of Dhole et al. 2021. "NL-Augmenter A Framework
    for Task-Sensitive Natural Language Augmentation". arXiv:2112.02721. with
    a reference to https://github.com/GEM-benchmark/NL-Augmenter/blob/1.0.0/
    transformations/city_names_transformation/transformation.py

    This maps each city name to a random city. All city names with the same
    surface text will be mapped to a same name unlike the original
    implementation.
    """

    name: str = "city_name_replacement"
    replacees_filename: str = "Eng_Pop.txt"
    replacers_filename: str = "Eng_Scarce.txt"

    def __init__(self, nlp=None):
        if nlp is None:
            try:
                self.nlp = spacy.load("en_core_web_sm", exclude=["tagger", "parser", "lemmatizer", "textcat"])
            except OSError:
                spacy.cli.download("en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm", exclude=["tagger", "parser", "lemmatizer", "textcat"])
        else:
            self.nlp = nlp
        data_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(data_dir, self.replacees_filename)) as fin:
            self.replacees: Set[str] = set(fin.read().split("\n"))
        with open(os.path.join(data_dir, self.replacers_filename)) as fin:
            self.replacers: List[str] = sorted(set(fin.read().split("\n")))

    def _is_target(self, ner_tag: str, text: str) -> bool:
        return (ner_tag == "GPE" or ner_tag == "LOC") and text in self.replacees

    @staticmethod
    def _span_overlap(spans) -> bool:
        return any((s1[1] > s2[0] for s1, s2 in pairwise(sorted(spans, key=lambda s: s[0]))))

    def perturb(self, text: str) -> str:
        doc = self.nlp(text)
        replaced_spans = [
            (doc[ent.start].idx, doc[ent.end - 1].idx + len(doc[ent.end - 1]), ent.text)
            for ent in doc.ents
            if self._is_target(ent.label_, ent.text)
        ]
        assert not self._span_overlap(replaced_spans)

        city_names = {span[2] for span in replaced_spans}
        if len(city_names) < len(self.replacers):
            replacers = random.sample(self.replacers, len(city_names))
        else:
            # in case there are more city names than our list
            replacers = random.choices(list(self.replacers), k=len(city_names))
        # map the same city names to a single city name unlike original impl.
        # make sure that it doesn't choose the same city name twice
        name_mapping = dict(zip(city_names, replacers))

        last_char_idx = 0
        text_fragments = []
        for span in sorted(replaced_spans, key=lambda s: s[0]):
            text_fragments.append(text[last_char_idx : span[0]])
            text_fragments.append(name_mapping[span[2]])
            last_char_idx = span[1]
        text_fragments.append(text[last_char_idx:])
        return "".join(text_fragments)
