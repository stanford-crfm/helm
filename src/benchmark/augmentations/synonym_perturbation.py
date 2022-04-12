from dataclasses import dataclass

import os
import nltk
from nltk.corpus import wordnet
from nltk.tokenize.treebank import TreebankWordDetokenizer
import numpy as np
import spacy

from benchmark.scenario import Instance
from .perturbation_description import PerturbationDescription
from .perturbation import Perturbation


@dataclass
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
        prob: float

    name: str = "SynonymPerturbation"

    def __init__(self, prob: float):
        self.prob: float = prob
        self.detokenizer = TreebankWordDetokenizer()

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
        return SynonymPerturbation.Description(self.name, self.prob)

    def synonyms_substitute(self, text):
        try:
            spacy_model = spacy.load("en_core_web_sm")
        except OSError:
            spacy.cli.download("en_core_web_sm")
            spacy_model = spacy.load("en_core_web_sm")

        upos_wn_dict = {
            "VERB": "v",
            "NOUN": "n",
            "ADV": "r",
            "ADJ": "s",
        }

        doc = spacy_model(text)

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
        result = self.detokenizer.detokenize(result)
        result = result.replace(" .", ".")

        return result

    def apply(self, instance: Instance, should_perturb_references: bool = True) -> Instance:
        assert instance.id is not None
        np.random.seed(int(instance.id[2:]))  # set seed based on instance ID
        return super().apply(instance, should_perturb_references)

    def perturb(self, text: str) -> str:
        return self.synonyms_substitute(text)
