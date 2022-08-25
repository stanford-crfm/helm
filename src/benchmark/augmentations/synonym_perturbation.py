from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Dict, List

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
            wordnet_key = f"{word}:{wordnet_pos}"
            assert wordnet_pos or (wordnet_key not in self.wordnet_synonyms)
            if wordnet_key in self.wordnet_synonyms:
                synonyms = self.wordnet_synonyms[wordnet_key]
                assert synonyms
                assert wordnet_pos
                synsets = wordnet.synsets(word, pos=wordnet_pos)
                synonyms2 = [lemma.name() for synset in synsets for lemma in synset.lemmas()]
                synonyms2 = [s for s in synonyms2 if s != word.lower()]
                synonyms2 = list(dict.fromkeys(synonyms2)) 
                if synonyms != synonyms2:
                    print("BOOM", wordnet_key, synonyms, synonyms2)
                    1/0
                randomness = self.random.uniform(0, 1)
                if randomness < self.prob:
                    synonym = self.random.choice(synonyms)
                    synonym = synonym.replace("_", " ")  # We might get synonyms such as "passive_voice"
                    word = match_case(word, synonym)
                if "Elmendorf" in text:
                    print(randomness, token.text, "->", word, wordnet_pos)
            perturbed_text += word + token.whitespace_

        return perturbed_text

    def apply(self, instance: Instance) -> Instance:
        assert instance.id is not None
        self.random = random.Random(int(instance.id[2:]))
        return super().apply(instance)

    def perturb(self, text: str) -> str:
        return self.synonyms_substitute(text)
