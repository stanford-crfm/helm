from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Dict, List
import os

import nltk
from nltk.corpus import wordnet
import spacy

from benchmark.scenarios.scenario import Instance
from common.general import match_case
from .perturbation_description import PerturbationDescription
from .perturbation import Perturbation


class SynonymPerturbation(Perturbation):
    """
    Synonyms. For implementation details, see
    https://github.com/GEM-benchmark/NL-Augmenter/blob/main/nlaugmenter/transformations/synonym_substitution/transformation.py
    This perturbation adds noise to a text source by randomly inserting synonyms of randomly selected
    words excluding punctuations and stopwords.
    The space of synonyms depends on WordNet and could be limited. The transformation might introduce
    non-grammatical segments.

    Perturbation example:
    Input:
        This was a good movie, would watch again.
    Output:
        This was a dependable movie, would determine again.
    """

    @dataclass(frozen=True)
    class Description(PerturbationDescription):
        prob: float = 0.0

    name: str = "synonym"

    def __init__(self, prob: float):
        # Assign parameters to instance variables
        self.prob: float = prob

        # Random generator specific to this class, will be set in the apply function
        self.random: random.Random

        # Initialize the model with spaCy: https://spacy.io/models/en
        try:
            self.spacy_model = spacy.load("en_core_web_sm")
        except OSError:
            # no idea why this keeps failing, tested multiple times
            spacy.cli.download("en_core_web_sm")  # type: ignore
            self.spacy_model = spacy.load("en_core_web_sm")

        try:
            _ = wordnet._morphy("test", "n")
        except LookupError:
            out_dir = "nltk_data"
            nltk.data.path.append(out_dir)
            if not os.path.exists(os.path.join(out_dir, "corpora/wordnet")):
                nltk.download("wordnet", download_dir=out_dir)
            if not os.path.exists(os.path.join(out_dir, "corpora/omw-1.4")):
                nltk.download("omw-1.4", download_dir=out_dir)

        wordnet_synonyms_path = Path(__file__).resolve().expanduser().parent / "wordnet_synonyms.json"
        with open(wordnet_synonyms_path) as f:
            self.wordnet_synonyms: Dict[str, List[str]] = json.load(f)

    @property
    def description(self) -> PerturbationDescription:
        return SynonymPerturbation.Description(name=self.name, robustness=True, prob=self.prob)

    def synonyms_substitute(self, text: str) -> str:
        spacy_to_wordnet_pos = {
            "VERB": "v",
            "NOUN": "n",
            "ADV": "r",
            "ADJ": "s",
        }

        doc = self.spacy_model(text)

        perturbed_text = ""
        for token in doc:
            word = token.text
            wordnet_pos = spacy_to_wordnet_pos.get(token.pos_)
            synonyms = []
            if wordnet_pos:
                for base in wordnet._morphy(word.lower(), wordnet_pos):  # _morphy returns the base form of a word
                    synonyms.extend(self.wordnet_synonyms.get(f"{base}:{wordnet_pos}", []))
            synonyms = [s for s in synonyms if s != word.lower()]
            synonyms = list(dict.fromkeys(synonyms))  # Make the list unique while preserving the order
            if synonyms:
                if self.random.uniform(0, 1) < self.prob:
                    synonym = self.random.choice(synonyms)
                    word = match_case(word, synonym)
            perturbed_text += word + token.whitespace_

        return perturbed_text

    def apply(self, instance: Instance) -> Instance:
        assert instance.id is not None
        self.random = random.Random(int(instance.id[2:]))
        return super().apply(instance)

    def perturb(self, text: str) -> str:
        return self.synonyms_substitute(text)
