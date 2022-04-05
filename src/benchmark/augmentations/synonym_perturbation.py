from dataclasses import dataclass


import re
import numpy as np
import spacy
from nltk.corpus import wordnet

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
        Albany Theatre is a historic theater in Albany, Georgia.
    Output:
        Albany Theatre is a historic dramaturgy in Albany, Georgia.

    Dependencies required to run this perturbation:
    This perturbation required WordNet and the Spacy model en_core_web_sm. Instructions to download both
    these models are specified below.

    1. Download WordNet by running the command

        python -m nltk.downloader wordnet omw-1.4

    2. Install the required spacy model by running the command

        pip install -U spacy
        python -m spacy download en_core_web_sm

    """

    @dataclass(frozen=True)
    class Description(PerturbationDescription):
        name: str
        prob: float

    name: str = "SynonymPerturbation"

    def __init__(self, prob: float):
        self.prob: float = prob

    @property
    def description(self) -> PerturbationDescription:
        return SynonymPerturbation.Description(self.name, self.prob)

    @staticmethod
    def untokenize(words):

        text = " ".join(words)
        step1 = text.replace("`` ", '"').replace(" ''", '"').replace(". . .", "...")
        step2 = step1.replace(" ( ", " (").replace(" ) ", ") ")
        step3 = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", step2)
        step4 = re.sub(r" ([.,:;?!%]+)$", r"\1", step3)
        step5 = step4.replace(" '", "'").replace(" n't", "n't").replace("can not", "cannot")
        step6 = step5.replace(" ` ", " '")
        return step6.strip()

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
        result = untokenize(result)  # noqa

        return result

    def apply(self, instance: Instance, should_perturb_references: bool = True) -> Instance:
        assert instance.id is not None
        np.random.seed(int(instance.id[2:]))  # set seed based on instance ID
        return super().apply(instance, should_perturb_references)

    def perturb(self, text: str) -> str:
        return self.synonyms_substitute(text)
