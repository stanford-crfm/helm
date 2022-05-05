from dataclasses import dataclass
import json
import random
from typing import Dict, List, Optional, Tuple

from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer

from benchmark.scenario import Instance
from common.general import match_case
from .perturbation_description import PerturbationDescription
from .perturbation import Perturbation


""" Gender term mappings """
GENDER_TERM_MAPPINGS: List[Tuple[str, ...]] = [
    # Inspired by Garg et al. (2018)
    ("child", "daughter", "son"),
    ("children", "daughters", "sons"),
    ("parent", "mother", "father"),
    ("parents", "mothers", "fathers"),
    ("kiddo", "girl", "boy"),
    ("kiddos", "girls", "boys"),
    ("person", "woman", "man"),
    ("people", "women", "men"),
    ("sibling", "daughter", "brother"),
    ("siblings", "daughters", "brothers"),
    ("nibling", "niece", "nephew"),
    ("niblings", "nieces", "nephews"),
    # Inspired by Bolukbasi et al. (2016)
    ("monarch", "queen", "king"),
    ("monarchs", "queens", "kings"),
    ("server", "waitress", "waiter"),
    ("servers", "waitresses", "waiters"),
    # Our additions
    ("everybody", "ladies", "guys"),
    ("everybody", "ladies", "guys"),
    ("salesperson", "saleswoman", "salesman"),
]

""" Gender pronoun mappings """
GENDER_PRONOUN_MAPPINGS: List[Tuple[str, ...]] = [
    ("he", "she", "they"),
    ("him", "her", "them"),
    ("his", "her", "their"),
    ("himself", "herself", "themselves"),
]


@dataclass
class GenderPerturbation(Perturbation):
    """ Individual fairness perturbation for gender terms and pronouns. """

    """ Short unique identifier of the perturbation (e.g., extra_space) """
    name: str = "gender_term"

    """ Line seperator """
    LINE_SEP = "\n"

    """ Genders defined by default """
    NEUTRAL = "neutral"
    FEMALE = "female"
    MALE = "male"
    GENDERS = [NEUTRAL, FEMALE, MALE]

    """ Modes """
    GENDER_TERM = "terms"
    GENDER_PRONOUN = "pronouns"
    MODES = [GENDER_TERM, GENDER_PRONOUN]
    MODE_TO_MAPPINGS = {GENDER_TERM: GENDER_TERM_MAPPINGS, GENDER_PRONOUN: GENDER_PRONOUN_MAPPINGS}

    @dataclass(frozen=True)
    class Description(PerturbationDescription):
        """ Description for the GenderPerturbation class. """

        name: str
        prob: float
        source_class: str
        target_class: str
        mapping_file_path: Optional[str]
        bidirectional: bool

    def __init__(
        self,
        mode: str,
        prob: float,
        source_class: str,
        target_class: str,
        mapping_file_path: Optional[str] = None,
        mapping_file_genders: List[str] = None,
        bidirectional: bool = False,
    ):
        """ Initialize the gender perturbation.

        Args:
            mode: The mode of the gender perturbation, should be one of
                "terms" or "pronouns".
            prob: Probability of substituting a word in the source class with
                a word in the target class given that a substitution is
                available.
            source_class: The source gender that will be substituted with
                the target gender.
            target_class: The target gender.
            mapping_file_path: The absolute path to a file containing the
                word mappings from the source gender to the target gender in
                a json format. The json dictionary must be of type
                List[List[str, ...]]. It is assumed that 0th index of the inner
                lists correspond to the 0th gender, 1st index to 1st gender
                and so on.
                If mapping_file_path is None, the default dictionary in
                self.MODE_TO_MAPPINGS for the provided source and target classes
                will be used, if available.
            mapping_file_genders: The genders in the mapping supplied in the
                mapping_file_path. The inner lists read from mapping_file_path
                should have the same length as the mapping_file_genders. The
                order of the genders is assumed to reflect the order in the
                mapping_file_path. Must not be None if mapping_file_path
                is set.
            bidirectional: Whether we should apply the perturbation in both
                directions. If we need to perturb a word, we first check if it
                is in list of source_class words, and replace it with the
                corresponding target_class word if so. If the word isn't in the
                source_class words, we check if it is in the target_class words,
                and replace it with the corresponding source_class word if so.
        """
        # self.random, will be set in the apply function
        self.random: random.Random

        assert mode in self.MODES
        self.mode = mode
        assert 0 <= prob <= 1
        self.prob = prob
        self.source_class: str = source_class
        self.target_class: str = target_class
        self.mapping_file_path: Optional[str] = mapping_file_path
        self.bidirectional: bool = bidirectional

        mappings: List[Tuple[str, ...]] = self.MODE_TO_MAPPINGS[self.mode]
        self.genders = self.GENDERS
        if self.mapping_file_path and mapping_file_genders:
            self.genders = mapping_file_genders
            with open(self.mapping_file_path, "r") as f:
                loaded_json = json.load(f)
                mappings = [tuple([str(e).lower() for e in t]) for t in loaded_json]
            assert mappings
        assert self.source_class in self.genders and self.target_class in self.genders
        # The min function is irrelevant, all we are doing is getting an element from the set
        assert all([len(m) == len(self.genders) for m in mappings])

        # Remove duplicates from the mappings list
        mappings = list(set(mappings))
        # Get source and target terms
        word_to_index: Dict[str, int] = {term: ind for ind, term in enumerate(self.genders)}
        word_lists = list(zip(*mappings))
        self.source_words: List[str] = list(word_lists[word_to_index[self.source_class]])
        self.target_words: List[str] = list(word_lists[word_to_index[self.target_class]])

        # Initialize the tokenizers
        self.tokenizer = TreebankWordTokenizer()
        self.detokenizer = TreebankWordDetokenizer()

    @property
    def description(self) -> PerturbationDescription:
        """ Return a perturbation description for this class. """
        return GenderPerturbation.Description(
            self.name, self.prob, self.source_class, self.target_class, self.mapping_file_path, self.bidirectional
        )

    def sample_word(self, word: str, source_words: List[str], target_words: List[str]) -> Optional[str]:
        """ Sample a word from target_terms if the word is in source_terms.

        Return None if the word wasn't in source_terms.
        """
        lowered_word = word.lower()
        if lowered_word in source_words:
            # Sample a new term and ensure that the case of the new word matches the original
            ind = source_words.index(lowered_word)
            synonym = target_words[ind]
            return match_case(word, synonym)
        return None

    def substitute_words(self, text: str) -> str:
        """ Substitute the terms of the source gender with those of the target gender. """
        lines, new_lines = text.split(self.LINE_SEP), []
        for line in lines:
            words, new_words = self.tokenizer.tokenize(line), []
            for word in words:
                if self.random.uniform(0, 1) < self.prob:
                    sampled_word = self.sample_word(word, self.source_words, self.target_words)
                    if not sampled_word and self.bidirectional:
                        sampled_word = self.sample_word(word, self.target_words, self.source_words)
                    word = sampled_word if sampled_word else word
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
        return self.substitute_words(text)
