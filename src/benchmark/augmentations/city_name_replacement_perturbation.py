import itertools
import os
import re
import random
from dataclasses import dataclass
from typing import List, Set

from .perturbation import Perturbation


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

    def __init__(self, allow_lower: bool):
        data_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(data_dir, self.replacees_filename)) as fin:
            replacees: Set[str] = set(fin.read().strip().split("\n"))
        if allow_lower:
            replacees |= {replacee.lower() for replacee in replacees}
        self.replacees = re.compile(r"\b(?:" + "|".join(f"(?:{re.escape(t)})" for t in replacees) + r")\b")
        # sort in descending order of length so that longer spans will match first
        with open(os.path.join(data_dir, self.replacers_filename)) as fin:
            self.replacers: List[str] = sorted(set(fin.read().split("\n")))
        self.allow_lower = allow_lower

    def perturb(self, text: str) -> str:
        replaced_spans = [(m.span()[0], m.span()[1], m.group(0)) for m in self.replacees.finditer(text)]

        city_names = {span[2] for span in replaced_spans}
        # this is only required when allow_lower=True, but do it anyway to simplify the code
        city_names_lower = {n.lower() for n in city_names}
        if len(city_names_lower) < len(self.replacers):
            replacers = random.sample(self.replacers, len(city_names_lower))
        else:
            # in case there are more city names than our list
            replacers = random.choices(self.replacers, k=len(city_names_lower))
        # map the same city names to a single city name unlike original impl.
        # sort city names for random seed consistency
        name_mapping_lower = dict(zip(sorted(city_names_lower), replacers))
        # this maps a city name in lower characters and in upper characters to the same city name
        # while keeping case the same as original
        name_mapping = {
            city_name: name_mapping_lower[city_name.lower()].lower()
            if city_name.lower() == city_name
            else name_mapping_lower[city_name.lower()]
            for city_name in city_names
        }

        last_char_idx = 0
        text_fragments = []
        for span in sorted(replaced_spans, key=lambda s: s[0]):
            text_fragments.append(text[last_char_idx : span[0]])
            text_fragments.append(name_mapping[span[2]])
            last_char_idx = span[1]
        text_fragments.append(text[last_char_idx:])
        return "".join(text_fragments)
