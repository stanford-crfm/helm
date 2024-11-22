from dataclasses import dataclass
import json
import os
from random import Random
from pathlib import Path
from typing import Dict, List

import nltk
from nltk.corpus import wordnet
import spacy

from helm.common.general import match_case, ensure_file_downloaded
from helm.benchmark.augmentations.perturbation_description import PerturbationDescription
from helm.benchmark.augmentations.perturbation import TextPerturbation
from helm.benchmark.runner import get_benchmark_output_path


class SynonymPerturbation(TextPerturbation):
    """
    Synonyms. For implementation details, see
    https://github.com/GEM-benchmark/NL-Augmenter/blob/main/nlaugmenter/transformations/synonym_substitution/transformation.py

    This perturbation adds noise to a text source by randomly inserting synonyms of randomly selected
    words excluding punctuations and stopwords.
    The space of synonyms depends on WordNet and could be limited. The transformation might introduce
    non-grammatical segments.

    Perturbation example:

    **Input:**
        This was a good movie, would watch again.

    **Output:**
        This was a dependable movie, would determine again.
    """

    @dataclass(frozen=True)
    class Description(PerturbationDescription):
        prob: float = 0.0

    name: str = "synonym"

    # For downloading wordnet_synonyms.json
    FILE_NAME: str = "wordnet_synonyms.json"
    SOURCE_URI: str = (
        "https://storage.googleapis.com/crfm-helm-public/source_datasets/"
        "augmentations/synonym_perturbation/wordnet_synonyms.json"
    )

    def __init__(self, prob: float):
        # Assign parameters to instance variables
        self.prob: float = prob

        # Initialize the model with spaCy: https://spacy.io/models/en
        try:
            self.spacy_model = spacy.load("en_core_web_sm")
        except OSError:
            spacy.cli.download("en_core_web_sm")  # type: ignore
            self.spacy_model = spacy.load("en_core_web_sm")

        output_dir = os.path.join(get_benchmark_output_path(), "perturbations", self.name)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        nltk.data.path.append(output_dir)
        try:
            # We cannot use wordnet.synsets directly since it's not thread-safe. So we copy the synsets to
            # wordnet_synonyms.json and use that in combination with _morphy (as done in the original wordnet.synsets).
            wordnet.ensure_loaded()
        except LookupError:
            if not os.path.exists(os.path.join(output_dir, "corpora/wordnet")):
                nltk.download("wordnet", download_dir=output_dir)
            if not os.path.exists(os.path.join(output_dir, "corpora/omw-1.4")):
                nltk.download("omw-1.4", download_dir=output_dir)
        wordnet.ensure_loaded()

        target_path = os.path.join(output_dir, self.FILE_NAME)
        ensure_file_downloaded(source_url=self.SOURCE_URI, target_path=target_path)
        with open(target_path) as f:
            self.wordnet_synonyms: Dict[str, List[str]] = json.load(f)

    @property
    def description(self) -> PerturbationDescription:
        return SynonymPerturbation.Description(name=self.name, robustness=True, prob=self.prob)

    def perturb(self, text: str, rng: Random) -> str:
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
            if synonyms and rng.uniform(0, 1) < self.prob:
                synonym = rng.choice(synonyms)
                word = match_case(word, synonym)
            perturbed_text += word + token.whitespace_

        return perturbed_text
