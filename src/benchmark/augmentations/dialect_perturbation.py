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
    "dude": ["n*ggah", "niggah", "manee", "n*gga", "nigga"],
    "huge": ["bigass"],
    "probably": ["prob", "prolly", "def", "probly", "deff"],
    "rad": ["dope"],
    "remember": ["rememba"],
    "screaming": ["screamin", "yellin", "hollering"],
    "sister": ["sista", "sis"],
    "these": ["dese", "dem"],
    "with": ["wit"],
}


@dataclass
class DialectPerturbation(Perturbation):
    """ Individual fairness perturbation for dialect. """

    """ Short unique identifier of the perturbation (e.g., extra_space) """
    name: str = "dialect"

    """ Line seperator """
    LINE_SEP = "\n"

    """ Dictionary mapping dialects to one another """
    SAE = "SAE"
    AAVE = "AAVE"
    MAPPING_DICTS = {
        (SAE, AAVE): SAE_TO_AAVE_MAPPING_DICT,
    }

    @dataclass(frozen=True)
    class Description(PerturbationDescription):
        """ Description for the DialectPerturbation class. """

        name: str
        prob: float
        source_class: str
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
        # self.random, will be set in the apply function
        self.random: random.Random

        assert 0 <= prob <= 1
        self.prob = prob
        self.source_class: str = source_class
        self.target_class: str = target_class

        self.mapping_file_path: Optional[str] = mapping_file_path
        if self.mapping_file_path:
            with open(self.mapping_file_path, "r") as f:
                loaded_json = json.load(f)
                mapping_dict = {k.lower(): [e.lower() for e in l] for k, l in loaded_json.items()}
        else:
            mapping = (self.source_class, self.target_class)
            if mapping not in self.MAPPING_DICTS:
                msg = f"""The mapping from the source class {self.source_class} to the
                          target class {self.target_class} isn't available in {self.MAPPING_DICTS}.
                       """
                raise ValueError(msg)
            mapping_dict = self.MAPPING_DICTS[mapping]
        self.mapping_dict: Dict[str, List[str]] = mapping_dict

        # Initialize the tokenizers
        self.tokenizer = TreebankWordTokenizer()
        self.detokenizer = TreebankWordDetokenizer()

    @property
    def description(self) -> PerturbationDescription:
        """ Return a perturbation description for this class. """
        return DialectPerturbation.Description(
            self.name, self.prob, self.source_class, self.target_class, self.mapping_file_path
        )

    def substitute_dialect(self, text: str) -> str:
        """ Substitute the source dialect in text with the target dialect. """
        lines, new_lines = text.split(self.LINE_SEP), []
        for line in lines:
            words, new_words = self.tokenizer.tokenize(line), []
            for word in words:
                lowered_word = word.lower()
                if lowered_word in self.mapping_dict and self.random.uniform(0, 1) < self.prob:
                    # Sample a new word and ensure that the case of the new word matches the original
                    synonym = self.random.choice(self.mapping_dict[lowered_word])
                    word = match_case(word, synonym)
                new_words.append(word)
            perturbed_line = str(self.detokenizer.detokenize(new_words))
            new_lines.append(perturbed_line)
        perturbed_text = self.LINE_SEP.join(new_lines)
        return perturbed_text

    def apply(self, instance: Instance, should_perturb_references: bool = True) -> Instance:
        """ Apply the perturbation to the provided instance. """
        assert instance.id is not None
        self.random = random.Random(int(instance.id[2:]))
        return super().apply(instance, should_perturb_references)

    def perturb(self, text: str) -> str:
        """ Perturb the provided text. """
        return self.substitute_dialect(text)
