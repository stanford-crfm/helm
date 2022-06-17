from dataclasses import dataclass
import os
from typing import List

import nltk
from nltk.corpus import wordnet
from nltk.tokenize.treebank import TreebankWordDetokenizer
import numpy as np
import spacy

from benchmark.scenario import Instance
from .perturbation_description import PerturbationDescription, ROBUSTNESS_TAG
from .perturbation import Perturbation


class SynonymPerturbation(Perturbation):
    """
    Synonyms. For implementation details, see
    https://github.com/GEM-benchmark/NL-Augmenter/blob/main/transformations/synonym_substitution
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
        name: str
        tags: List[str]
        prob: float

    name: str = "SynonymPerturbation"
    tags: List[str] = [ROBUSTNESS_TAG]

    def __init__(self, prob: float):
        self.prob: float = prob
        self.detokenizer = TreebankWordDetokenizer()

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
        return SynonymPerturbation.Description(self.name, self.tags, self.prob)

    def synonyms_substitute(self, text: str) -> str:
        upos_wn_dict = {
            "VERB": "v",
            "NOUN": "n",
            "ADV": "r",
            "ADJ": "s",
        }

        doc = self.spacy_model(text)

        result = []

        for token in doc:
            word = token.text
            wn_pos = upos_wn_dict.get(token.pos_)
            if wn_pos is None:
                result.append(word)
            else:
                syns = wordnet.synsets(word, pos=wn_pos)
                syns = [syn.name().split(".")[0] for syn in syns]
                syns = [syn for syn in syns if syn.lower() != word.lower()]
                if len(syns) > 0 and np.random.random() < self.prob:
                    result.append(np.random.choice(syns).replace("_", " "))
                else:
                    result.append(word)

        # detokenize sentences
        perturbed_text = self.detokenizer.detokenize(result).replace(" .", ".")

        return perturbed_text

    def apply(self, instance: Instance, should_perturb_references: bool = True) -> Instance:
        assert instance.id is not None
        np.random.seed(int(instance.id[2:]))  # set seed based on instance ID
        return super().apply(instance, should_perturb_references)

    def perturb(self, text: str) -> str:
        return self.synonyms_substitute(text)
