from dataclasses import dataclass
import json
import random
from typing import Dict, Optional, List

from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer

from benchmark.scenario import Instance
from common.general import match_case
from .perturbation_description import PerturbationDescription
from .perturbation import Perturbation


""" Dictionary mapping words in SAE to their counterparts in AAVE.

List taken from from Ziems et al. (2022): https://arxiv.org/abs/2204.03031
"""
SAE_TO_AAVE_MAPPING_DICT = {
    "arguing": ["beefing", "beefin", "arguin"],
    "anymore": ["nomore", "nomo"],
    "brother": ["homeboy"],
    "classy": ["fly"],
    "dude": ["n*ggah", "manee", "n*gga"],
    "huge": ["bigass"],
    "probably": ["prob", "prolly", "def", "probly", "deff"],
    "rad": ["dope"],
    "remember": ["rememba"],
    "screaming": ["screaming", "yellin", "hollering"],
    "sister": ["sista", "sis"],
    "these": ["dese", "dem"],
    "with": ["wit"],
}


@dataclass
class DialectPerturbation(Perturbation):
    """ Individual fairness perturbation for dialect. """

    """ Random seed """
    SEED = 1885

    """ Line seperator """
    LINE_SEP = "\n"

    """ Dictionary mapping dialects to one another """
    SAE = "SAE"
    AAVE = "AAVE"
    MAPPING_DICTS = {
        (SAE, AAVE): SAE_TO_AAVE_MAPPING_DICT,
    }

    """ Short unique identifier of the perturbation (e.g., extra_space) """
    name: str = "dialect"

    @dataclass(frozen=True)
    class Description(PerturbationDescription):
        """ Description for the DialectPerturbation class. """
        name: str
        prob: float
        original_class: str
        target_class: str
        mapping_file_path: Optional[str]

    def __init__(self, prob: float, source_class: str, target_class: str, mapping_file_path: Optional[str] = None):
        """ Initialize the dialect perturbation.

        If mapping_file_path is not provided, (source_class, target_class)
        should be ("SAE", "AAVE").

        Args:
            prob: Probability of substituting a word in the original class with
                a word in the target class given that a substitution is
                available.
            source_class: The source dialect that will be substituted with
                the target dialect.
            target_class: The target dialect.
            mapping_file_path: The absolute path to a file containing the
                word mappings from the source dialect to the target dialect in
                a json format. The json dictionary must be of type
                Dict[str, List[str]]. Otherwise, the default dictionary in
                self.MAPPING_DICTS for the provided source and target classes
                will be used, if available.
        """
        assert 0 <= prob <= 1
        self.prob = prob
        self.original_class: str = source_class
        self.target_class: str = target_class

        self.mapping_file_path: Optional[str] = mapping_file_path
        if mapping_file_path:
            with open(mapping_file_path, "r") as f:
                self.mapping_dict: Dict[str, List[str]] = json.load(f)
        else:
            mapping = (source_class, target_class)
            assert mapping in self.MAPPING_DICTS
            self.mapping_dict = self.MAPPING_DICTS[mapping]

        # Initialize the tokenizers
        self.tokenizer = TreebankWordTokenizer()
        self.detokenizer = TreebankWordDetokenizer()

        # Random generator for our perturbation
        self.random: random.Random = random.Random(self.SEED)

    @property
    def description(self) -> PerturbationDescription:
        """ Return a perturbation description for this class. """
        return DialectPerturbation.Description(
            self.name, self.prob, self.original_class, self.target_class, self.mapping_file_path
        )

    def substitute_dialect(self, text: str) -> str:
        """ Substitute the source dialect in text with the target dialect. """
        lines, new_lines = text.split(self.LINE_SEP), []
        for line in lines:
            words, new_words = self.tokenizer.tokenize(line), []
            for word in words:
                if word.lower() in self.mapping_dict and self.random.uniform(0, 1) < self.prob:
                    # Sample a new word and ensure that the case of the new word matches the original
                    new_word = self.random.choice(self.mapping_dict[word.lower()])
                    word = match_case(word, new_word)
                new_words.append(word)
            perturbed_line = str(self.detokenizer.detokenize(new_words))
            new_lines.append(perturbed_line)
        perturbed_text = self.LINE_SEP.join(new_lines)
        return perturbed_text

    def apply(self, instance: Instance, should_perturb_references: bool = True) -> Instance:
        """ Apply the perturbation to the provided instance. """
        assert instance.id is not None
        return super().apply(instance, should_perturb_references)

    def perturb(self, text: str) -> str:
        """ Perturb the provided text. """
        return self.substitute_dialect(text)
