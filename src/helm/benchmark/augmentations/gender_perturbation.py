from dataclasses import dataclass
import json
from random import Random
import re
from typing import Dict, List, Optional, Tuple

from helm.common.general import match_case
from helm.benchmark.augmentations.perturbation_description import PerturbationDescription
from helm.benchmark.augmentations.perturbation import TextPerturbation


""" Gender term mappings """
GENDER_TERM_MAPPINGS: List[Tuple[str, ...]] = [
    # List based on Garg et al. (2018)
    ("child", "daughter", "son"),
    ("children", "daughters", "sons"),
    ("parent", "mother", "father"),
    ("parents", "mothers", "fathers"),
    ("kiddo", "girl", "boy"),
    ("kiddos", "girls", "boys"),
    ("person", "woman", "man"),
    ("people", "women", "men"),
    ("sibling", "sister", "brother"),
    ("siblings", "sisters", "brothers"),
    ("nibling", "niece", "nephew"),
    ("niblings", "nieces", "nephews"),
    # List based on Bolukbasi et al. (2016)
    ("monarch", "queen", "king"),
    ("monarchs", "queens", "kings"),
    ("server", "waitress", "waiter"),
    ("servers", "waitresses", "waiters"),
    # Our additions
    ("parent", "mom", "dad"),
    ("parents", "moms", "dads"),
    ("stepchild", "stepdaughter", "stepson"),
    ("stepchildren", "stepdaughters", "stepsons"),
    ("stepparent", "stepmother", "stepfather"),
    ("stepparents", "stepmothers", "stepfathers"),
    ("stepparent", "stepmom", "stepdad"),
    ("stepparents", "stepmoms", "stepdads"),
    ("grandchild", "granddaughter", "grandson"),
    ("grandchildren", "granddaughters", "grandsons"),
    ("grandparent", "grandmother", "grandfather"),
    ("grandparents", "grandmothers", "grandfather"),
    ("grandparent", "grandma", "granddad"),
    ("grandparents", "grandmas", "granddads"),
    ("human", "female", "male"),
    ("humans", "females", "males"),
]

""" Gender pronoun mappings """
# The overlaps between the pairs cause our replacements to be wrong in certain
# cases (direct pronouns vs. indirect pronouns). In these cases, we keep the
# first match instead of making our decision using a POS tagger for simplicity
GENDER_PRONOUN_MAPPINGS: List[Tuple[str, ...]] = [
    # List from Lauscher et. al. 2022
    ("they", "she", "he"),
    ("them", "her", "him"),
    ("their", "her", "his"),
    ("theirs", "hers", "his"),
    ("themselves", "herself", "himself"),
]


class GenderPerturbation(TextPerturbation):
    """Individual fairness perturbation for gender terms and pronouns."""

    """ Short unique identifier of the perturbation (e.g., extra_space) """
    name: str = "gender"

    should_perturb_references: bool = True

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
        """Description for the GenderPerturbation class."""

        mode: str = ""
        prob: float = 0.0
        source_class: str = ""
        target_class: str = ""
        bidirectional: bool = False

    def __init__(
        self,
        mode: str,
        prob: float,
        source_class: str,
        target_class: str,
        mapping_file_path: Optional[str] = None,
        mapping_file_genders: Optional[List[str]] = None,
        bidirectional: bool = False,
    ):
        """Initialize the gender perturbation.

        Args:
            mode: The mode of the gender perturbation, must be one of
                "terms" or "pronouns".
            prob: Probability of substituting a word in the source class with
                a word in the target class given that a substitution is
                available.
            source_class: The source gender that will be substituted with
                the target gender. If mapping_file_path is provided, the source
                class must be one of the genders in it. If not, it must be
                exactly one of `male`, `female`, and `neutral. Case-insensitive.
            target_class: Same as the source class, but for the target gender.
            mapping_file_path: The absolute path to a file containing the
                word mappings from the source gender to the target gender in
                a json format. The json dictionary must be of type
                List[List[str, ...]]. It is assumed that 0th index of the inner
                lists correspond to the 0th gender, 1st index to 1st gender
                and so on. All word cases are lowered.
                If mapping_file_path is None, the default dictionary in
                self.MODE_TO_MAPPINGS for the provided source and target classes
                will be used, if available.
            mapping_file_genders: The genders in the mapping supplied in the
                mapping_file_path. The inner lists read from mapping_file_path
                should have the same length as the mapping_file_genders. The
                order of the genders is assumed to reflect the order in the
                mapping_file_path. Must not be None if mapping_file_path
                is set. All word cases are lowered.
            bidirectional: Whether we should apply the perturbation in both
                directions. If we need to perturb a word, we first check if it
                is in list of source_class words, and replace it with the
                corresponding target_class word if so. If the word isn't in the
                source_class words, we check if it is in the target_class words,
                and replace it with the corresponding source_class word if so.
        """
        # Assign parameters to instance variables
        assert mode in self.MODES
        self.mode = mode

        assert 0 <= prob <= 1
        self.prob = prob

        self.source_class: str = source_class.lower()
        self.target_class: str = target_class.lower()
        self.mapping_file_path: Optional[str] = mapping_file_path
        self.bidirectional: bool = bidirectional

        # Get mappings and self.genders
        mappings: List[Tuple[str, ...]] = self.MODE_TO_MAPPINGS[self.mode]
        self.genders = self.GENDERS
        if self.mapping_file_path and mapping_file_genders:
            mappings = self.load_mappings(self.mapping_file_path)
            self.genders = [g.lower() for g in mapping_file_genders]
        assert mappings and self.source_class in self.genders and self.target_class in self.genders
        assert all([len(m) == len(self.genders) for m in mappings])

        # Get source and target words
        gender_to_ind: Dict[str, int] = {gender: ind for ind, gender in enumerate(self.genders)}
        word_lists = list(zip(*mappings))
        self.source_words: List[str] = list(word_lists[gender_to_ind[self.source_class]])
        self.target_words: List[str] = list(word_lists[gender_to_ind[self.target_class]])

        # Get word_synonym_pairs
        self.word_synonym_pairs = list(zip(self.source_words, self.target_words))

        # If self.bidirectional flag is set, extend the pairs list
        if self.bidirectional:
            new_pairs = list(zip(self.target_words, self.source_words))
            self.word_synonym_pairs.extend(new_pairs)

    @property
    def description(self) -> PerturbationDescription:
        """Return a perturbation description for this class."""
        return GenderPerturbation.Description(
            name=self.name,
            mode=self.mode,
            fairness=True,
            prob=self.prob,
            source_class=self.source_class,
            target_class=self.target_class,
            bidirectional=self.bidirectional,
        )

    @staticmethod
    def load_mappings(mapping_file_path: str) -> List[Tuple[str, ...]]:
        """Load mappings as a list."""
        with open(mapping_file_path, "r") as f:
            loaded_json = json.load(f)
            return [tuple([str(e).lower() for e in t]) for t in loaded_json]

    def substitute_word(self, text: str, word: str, synonym: str, rng: Random) -> str:
        """Substitute the occurences of word in text with its synonym with self.probability"""
        # Pattern capturing any occurence of given word in the text, surrounded by non-alphanumeric characters
        pattern = f"[^\\w]({word})[^\\w]"

        # Substitution function
        def sub_func(m: re.Match):
            match_str = m.group(0)  # The full match (e.g. " Man ", " Man,", " Man.", "-Man.")
            match_word = m.group(1)  # Captured group (e.g. "Man")
            if rng.uniform(0, 1) < self.prob:
                syn = match_case(match_word, synonym)  # Synoynm with matching case (e.g. "Woman")
                match_str = match_str.replace(
                    match_word, syn
                )  # Synonym placed in the matching group (e.g. " Woman ", " Woman,", " Woman.", "-Woman")
            return match_str

        # Execute the RegEx
        return re.sub(pattern, sub_func, text, flags=re.IGNORECASE)

    def perturb(self, text: str, rng: Random) -> str:
        """Perform the perturbations on the provided text."""
        # Substitute the words
        for word, synonym in self.word_synonym_pairs:
            text = self.substitute_word(text, word, synonym, rng)

        return text
