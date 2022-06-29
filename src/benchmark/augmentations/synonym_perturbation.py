from dataclasses import dataclass
import os
import random

import nltk
from nltk.corpus import wordnet
import spacy

from benchmark.scenario import Instance
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

    name: str = "SynonymPerturbation"

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
            _ = wordnet.synsets("test")
        except LookupError:
            # TODO get directory from broader output_path
            out_dir = "nltk_data"
            nltk.data.path.append(out_dir)
            if not os.path.exists(os.path.join(out_dir, "corpora/wordnet")):
                nltk.download("wordnet", download_dir=out_dir)
            if not os.path.exists(os.path.join(out_dir, "corpora/omw-1.4")):
                nltk.download("omw-1.4", download_dir=out_dir)

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
            if wordnet_pos:
                synsets = wordnet.synsets(word, pos=wordnet_pos)
                synonyms = [synset.name().split(".")[0] for synset in synsets]
                synonyms = [s for s in synonyms if s != word.lower()]
                if synonyms and self.random.uniform(0, 1) < self.prob:
                    synonym = self.random.choice(synonyms)
                    synonym = synonym.replace("_", " ")  # We might get synonyms such as "passive_voice"
                    word = match_case(word, synonym)
            perturbed_text += word + token.whitespace_

        return perturbed_text

    def apply(self, instance: Instance) -> Instance:
        assert instance.id is not None
        self.random = random.Random(int(instance.id[2:]))
        return super().apply(instance)

    def perturb(self, text: str) -> str:
        return self.synonyms_substitute(text)
