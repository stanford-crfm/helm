from dataclasses import dataclass, replace
import json
import os
from random import Random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional

from helm.common.general import ensure_file_downloaded, ensure_directory_exists
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.benchmark.scenarios.scenario import Input, Instance, Reference, Output
from helm.benchmark.augmentations.perturbation_description import PerturbationDescription
from helm.benchmark.augmentations.perturbation import Perturbation, TextPerturbation
from helm.benchmark.runner import get_benchmark_output_path


############################################################


class ChineseTyposPerturbation(TextPerturbation):
    """
    Chinese typos. For implementation details, see
    https://github.com/GEM-benchmark/NL-Augmenter/tree/main/nlaugmenter/transformations/chinese_butter_fingers_perturbation

    This perturbation adds noise to a text source by randomly replacing Chinese characters or words by
    other characters or words that share a similar Pinyin.

    Perturbation example:

    **Input:**
        我想买一部新手机。

    **Output:**
        我想买一部新收集。
    """

    @dataclass(frozen=True)
    class Description(PerturbationDescription):
        prob: float = 0.0
        rare_char_prob: float = 0.05
        consider_tone: bool = False
        word_level_perturb: bool = True

    name: str = "chinese_typos"

    # For downloading resources
    ASSET_URL = "http://39.108.215.175/assets/butter_finger"
    FILE_NAMES: List[str] = [
        "pinyin_to_char.json",
        "toneless_pinyin_to_char.json",
        "pinyin_to_common_char.json",
        "toneless_pinyin_to_common_char.json",
        "pinyin_to_word.json",
        "toneless_pinyin_to_word.json",
    ]

    def __init__(
        self,
        prob: float,
        rare_char_prob: float = 0.05,
        consider_tone: bool = False,
        word_level_perturb: bool = True,
    ):
        # Assign parameters to instance variables
        self.prob: float = prob
        self.rare_char_prob: float = rare_char_prob  # How likely we will use rare Chinese characters
        self.consider_tone: bool = (
            consider_tone  # Should we take the tone of Pinyin into account when considering similar char/words
        )
        self.word_level_perturb: bool = word_level_perturb  # Whether we perturb text on the character or word level

        # Ensure all necessary data are downloaded
        output_dir = os.path.join(get_benchmark_output_path(), "perturbations", self.name)
        ensure_directory_exists(output_dir)
        for filename in self.FILE_NAMES:
            target_path = os.path.join(output_dir, filename)
            SOURCE_URL: str = f"{self.ASSET_URL}/{filename}"
            ensure_file_downloaded(source_url=SOURCE_URL, target_path=target_path)

        # Load the data for the perturbation
        with open(
            os.path.join(
                output_dir,
                "pinyin_to_char.json" if self.consider_tone else "toneless_pinyin_to_char.json",
            )
        ) as f:
            self.chinese_character_database: Dict[str, List[str]] = json.load(f)
        with open(
            os.path.join(
                output_dir,
                "pinyin_to_common_char.json" if self.consider_tone else "toneless_pinyin_to_common_char.json",
            )
        ) as f:
            self.common_chinese_character_database: Dict[str, List[str]] = json.load(f)
        with open(
            os.path.join(
                output_dir,
                "pinyin_to_word.json" if self.consider_tone else "toneless_pinyin_to_word.json",
            )
        ) as f:
            self.chinese_words_database: Dict[str, List[str]] = json.load(f)

    @property
    def description(self) -> PerturbationDescription:
        return ChineseTyposPerturbation.Description(
            name=self.name,
            robustness=True,
            prob=self.prob,
            rare_char_prob=self.rare_char_prob,
            consider_tone=self.consider_tone,
            word_level_perturb=self.word_level_perturb,
        )

    def perturb(self, text: str, rng: Random) -> str:
        try:
            import jieba
        except ModuleNotFoundError as e:
            handle_module_not_found_error(e, ["cleva"])
        butter_text: str = ""
        output: List[str] = jieba.lcut(text)
        if self.word_level_perturb:
            words_to_similar_word_dict = self.get_words_with_similar_pinyin(
                output,
                self.rare_char_prob,
                self.chinese_character_database,
                self.common_chinese_character_database,
                self.chinese_words_database,
                self.consider_tone,
                rng,
            )
            for word in output:
                similar_pinyin_words = words_to_similar_word_dict[word]
                if rng.random() <= self.prob and len(similar_pinyin_words) != 0:
                    new_chinese_character = rng.choice(similar_pinyin_words)
                else:
                    new_chinese_character = word
                butter_text += new_chinese_character
        else:
            for chinese_character in text:
                similar_pinyins = self.get_characters_with_similar_pinyin(
                    chinese_character,
                    self.rare_char_prob,
                    self.chinese_character_database,
                    self.common_chinese_character_database,
                    self.consider_tone,
                    rng,
                )
                if rng.random() <= self.prob and similar_pinyins != "":
                    new_chinese_character = rng.choice(similar_pinyins)
                else:
                    new_chinese_character = chinese_character

                butter_text += new_chinese_character
        return butter_text

    def get_characters_with_similar_pinyin(
        self,
        chinese_character: str,
        rare_word_prob: float,
        chinese_character_database: Dict[str, List[str]],
        common_chinese_character_database: Dict[str, List[str]],
        consider_tone: bool,
        rng: Random,
    ) -> str:
        try:
            import pypinyin
        except ModuleNotFoundError as e:
            handle_module_not_found_error(e, ["cleva"])
        pinyin_for_char_to_be_perturbed: str = "".join(
            [item for pinyin in pypinyin.pinyin(chinese_character) for item in pinyin]
        )

        chars_with_similar_pinyin = ""
        if rng.random() <= rare_word_prob:
            chars_with_similar_pinyin = self.retrieve_from_database(
                chinese_character,
                chars_with_similar_pinyin,
                chinese_character_database,
                consider_tone,
                pinyin_for_char_to_be_perturbed,
            )
        else:
            chars_with_similar_pinyin = self.retrieve_from_database(
                chinese_character,
                chars_with_similar_pinyin,
                common_chinese_character_database,
                consider_tone,
                pinyin_for_char_to_be_perturbed,
            )

        return chars_with_similar_pinyin

    def get_words_with_similar_pinyin(
        self,
        text: List[str],
        rare_word_prob: float,
        chinese_character_database: Dict[str, List[str]],
        common_chinese_character_database: Dict[str, List[str]],
        chinese_words_database: Dict[str, List[str]],
        consider_tone: bool,
        rng: Random,
    ) -> Dict[str, List[str]]:
        words_to_similar_word_dict: Dict[str, List[str]] = {}
        for original_word in text:
            words_to_similar_word_dict[original_word] = self.get_similar_word_pinyin_list(
                chinese_character_database,
                chinese_words_database,
                common_chinese_character_database,
                consider_tone,
                original_word,
                rare_word_prob,
                rng,
            )
        return words_to_similar_word_dict

    def get_similar_word_pinyin_list(
        self,
        chinese_character_database: Dict[str, List[str]],
        chinese_words_database: Dict[str, List[str]],
        common_chinese_character_database: Dict[str, List[str]],
        consider_tone: bool,
        original_word: str,
        rare_word_prob: float,
        rng: Random,
    ) -> List[str]:
        try:
            import unidecode
            import pypinyin
        except ModuleNotFoundError as e:
            handle_module_not_found_error(e, ["cleva"])
        if len(original_word) == 1:
            similar_pinyins = self.get_characters_with_similar_pinyin(
                original_word,
                rare_word_prob,
                chinese_character_database,
                common_chinese_character_database,
                consider_tone,
                rng,
            )
            similar_word_pinyin_list = [char for char in similar_pinyins]
        elif len(original_word) > 1:
            original_word_pinyins = pypinyin.pinyin(original_word)
            original_word_pinyins_flatten = [item for pinyin in original_word_pinyins for item in pinyin]
            original_word_pinyins_string = "".join(original_word_pinyins_flatten)
            if not consider_tone:
                original_word_pinyins_string = unidecode.unidecode(original_word_pinyins_string)
            candidate_words = chinese_words_database.get(original_word_pinyins_string, [])
            similar_word_pinyin_list = []
            for word in candidate_words:
                if word != original_word:
                    similar_word_pinyin_list.append(word)
        return similar_word_pinyin_list

    def retrieve_from_database(
        self,
        chinese_character: str,
        chars_with_similar_pinyin: str,
        chinese_character_database: Dict[str, List[str]],
        consider_tone: bool,
        pinyin_for_char_to_be_perturbed: str,
    ) -> str:
        try:
            import unidecode
        except ModuleNotFoundError as e:
            handle_module_not_found_error(e, ["cleva"])
        if not consider_tone:
            pinyin_for_char_to_be_perturbed = unidecode.unidecode(pinyin_for_char_to_be_perturbed)
        candidate_chars = chinese_character_database.get(pinyin_for_char_to_be_perturbed, [])
        for char in candidate_chars:
            if chinese_character != char:
                chars_with_similar_pinyin += char
        return chars_with_similar_pinyin


class ChineseSynonymPerturbation(TextPerturbation):
    """
    Chinese synonyms. For implementation details, see
    https://github.com/GEM-benchmark/NL-Augmenter/blob/main/nlaugmenter/transformations/chinese_antonym_synonym_substitution

    This perturbation adds noise to a text source by randomly inserting synonyms of randomly selected
    words excluding punctuations and stopwords.

    Perturbation example:

    **Input:**
        裸婚，这里的“裸”，指物质财富匮乏的情况下结婚，例如：无房无车无存款，有时候用于强调现实的无奈，也有时候用于强调人对情感的关注。

    **Output:**
        裸婚，这里底“裸”，指物质财富匮乏的情况下结婚，譬如说：无房无车无储蓄，有时候用于强调现实的无奈，亦有时候用来强调人士对情感的关注。
    """

    @dataclass(frozen=True)
    class Description(PerturbationDescription):
        prob: float = 0.0
        trial_num: int = 10

    name: str = "chinese_synonym"

    # For downloading resources
    SOURCE_URL: str = "http://39.108.215.175/assets/synonyms.json"

    def __init__(self, prob: float, trial_num: int = 10):
        # Assign parameters to instance variables
        self.prob: float = prob
        self.trial_num: int = trial_num  # Number of trial to get a 100% perturbed text

        target_dir = os.path.join(get_benchmark_output_path(), "perturbations", self.name, "synonyms.json")
        ensure_directory_exists(os.path.dirname(target_dir))
        ensure_file_downloaded(source_url=self.SOURCE_URL, target_path=target_dir)
        with open(os.path.join(target_dir)) as f:
            self.synonym_dict: Dict[str, List[str]] = json.load(f)

    @property
    def description(self) -> PerturbationDescription:
        return ChineseSynonymPerturbation.Description(
            name=self.name, robustness=True, prob=self.prob, trial_num=self.trial_num
        )

    def perturb(self, text: str, rng: Random) -> str:
        try:
            import jieba
        except ModuleNotFoundError as e:
            handle_module_not_found_error(e, ["cleva"])
        words = jieba.lcut(text)

        for _ in range(self.trial_num):
            perturbed_text = ""
            for w in words:
                if (w in self.synonym_dict) and rng.random() < self.prob:
                    perturbed_text += self.sample_word(self.synonym_dict[w], rng)
                else:
                    perturbed_text += w

            if perturbed_text != text:
                break

        return perturbed_text

    def sample_word(self, sample_list: List[str], rng: Random) -> str:
        index = rng.randint(0, len(sample_list) - 1)
        return sample_list[index]


class CLEVAMildMixPerturbation(TextPerturbation):
    """
    CLEVA robustness perturbation that composes several perturbations.
    """

    name: str = "cleva_mild_mix"

    # Don't perturb references because it's not fair to have to generate broken text.
    should_perturb_references: bool = False

    def __init__(self):
        self.synonym_perturbation = ChineseSynonymPerturbation(0.3)
        self.chinese_typos_perturbation = ChineseTyposPerturbation(0.05)

    @property
    def description(self) -> PerturbationDescription:
        return PerturbationDescription(name=self.name, robustness=True)

    def perturb(self, text: str, rng: Random) -> str:
        # Original CLEVA paper additionally adopts the "character swapping",
        # but we find that it has a negative impact on many reasoning
        # tasks. Therefore, we do not include it here.
        text = self.synonym_perturbation.perturb(text, rng)
        text = self.chinese_typos_perturbation.perturb(text, rng)
        return text


############################################################


class ChineseGenderPerturbation(TextPerturbation):
    """Individual fairness perturbation for Chinese gender terms and pronouns."""

    name: str = "chinese_gender"

    should_perturb_references: bool = True

    """ Genders defined by default """
    FEMALE = "female"
    MALE = "male"
    GENDERS = [FEMALE, MALE]

    """ Modes """
    GENDER_TERM = "terms"
    GENDER_PRONOUN = "pronouns"
    MODES = [GENDER_TERM, GENDER_PRONOUN]

    """ Resources """
    SOURCE_URL: str = "http://39.108.215.175/assets/gender_term.txt"

    @dataclass(frozen=True)
    class Description(PerturbationDescription):
        """Description for the GenderPerturbation class."""

        mode: str = ""
        prob: float = 0.0
        source_class: str = ""
        target_class: str = ""

    def __init__(
        self,
        mode: str,
        prob: float,
        source_class: str,
        target_class: str,
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
                exactly one of `male`, `female`, and `neutral`. Case-insensitive.
            target_class: Same as the source class, but for the target gender.
        """
        # Assign parameters to instance variables
        assert mode in self.MODES
        self.mode = mode

        assert 0 <= prob <= 1
        self.prob = prob

        self.source_class: str = source_class.lower()
        self.target_class: str = target_class.lower()

        if self.mode == self.GENDER_TERM:
            self.term_dict: Dict[Tuple[str, str], Dict[str, str]] = defaultdict(dict)

            target_path = os.path.join(get_benchmark_output_path(), "perturbations", self.name, "gender_term.txt")
            ensure_directory_exists(os.path.dirname(target_path))
            ensure_file_downloaded(source_url=self.SOURCE_URL, target_path=target_path)
            with open(target_path) as fin:
                for line in fin.readlines():
                    splits: List[str] = line.strip("\n").split(" ")
                    self.term_dict[(self.MALE, self.FEMALE)][splits[0]] = splits[1]
                    self.term_dict[(self.FEMALE, self.MALE)][splits[1]] = splits[0]
        elif self.mode == self.GENDER_PRONOUN:
            self.term_dict = {
                (self.MALE, self.FEMALE): {
                    "他": "她",
                },
                (self.FEMALE, self.MALE): {
                    "她": "他",
                },
            }

    @property
    def description(self) -> PerturbationDescription:
        """Return a perturbation description for this class."""
        return ChineseGenderPerturbation.Description(
            name=self.name,
            mode=self.mode,
            fairness=True,
            prob=self.prob,
            source_class=self.source_class,
            target_class=self.target_class,
        )

    def perturb(self, text: str, rng: Random) -> str:
        """Perform the perturbations on the provided text."""
        try:
            import jieba
        except ModuleNotFoundError as e:
            handle_module_not_found_error(e, ["cleva"])
        words = jieba.lcut(text)

        mapping_dict = self.term_dict[(self.source_class, self.target_class)]
        perturbed_text = ""
        for w in words:
            if w in mapping_dict and rng.random() < self.prob:
                perturbed_text += mapping_dict[w]
            else:
                perturbed_text += w

        return perturbed_text


class ChinesePersonNamePerturbation(Perturbation):
    """Individual fairness perturbation for Chinese person names."""

    """ Short unique identifier of the perturbation (e.g., extra_space) """
    name: str = "chinese_person_name"

    should_perturb_references: bool = True

    """ Resources """
    SOURCE_URL: str = "http://39.108.215.175/assets/chinese_name_gender.json"
    OUTPUT_PATH = os.path.join(get_benchmark_output_path(), "perturbations", name)

    """ Gender categories """
    GENDER_CATEGORY = "gender"
    FEMALE = "female"
    MALE = "male"
    GENDERS = [FEMALE, MALE]

    @dataclass(frozen=True)
    class Description(PerturbationDescription):
        """Description for the ChinesePersonNamePerturbation class.

        Explanation for the fields are provided in the docstring of
        ChinesePersonNamePerturbation.__init__, except source_class and target_class
        fields, which correspond to the string representation of the
        corresponding parameters passed to __init__.
        """

        prob: float = 0.0
        source_class: str = ""
        target_class: str = ""
        preserve_gender: bool = False

    def __init__(
        self,
        prob: float,
        source_class: Dict[str, str],
        target_class: Dict[str, str],
        preserve_gender: bool = True,
    ):
        """Chinese person name perturbation. For implementation details, see
        https://github.com/GEM-benchmark/NL-Augmenter/tree/main/nlaugmenter/transformations/chinese_person_named_entities_gender

        Code adopted from
        https://github.com/stanford-crfm/helm/blob/main/src/helm/benchmark/augmentations/person_name_perturbation.py

        Args:
            prob: Probability of substituting a word in the source class with
                a word in the target class given that a substitution is
                available.
            source_class: The properties of the source class. The keys of the
                dictionary should correspond to categories ("gender" only for
                now) and the values should be the corresponding values. If
                more than one category is provided. Case-insensitive.
            target_class: Same as source_class, but specifies the target_class.
            preserve_gender: If set to True, we preserve the gender when
                mapping names of one category to those of another. If we can't
                find the gender association for a source_word, we randomly
                pick from one of the target names.
        """
        self.output_path: str = self.OUTPUT_PATH
        Path(self.output_path).mkdir(parents=True, exist_ok=True)

        # Assign parameters to instance variables
        assert 0 <= prob <= 1
        self.prob = prob

        self.source_class: Dict[str, str] = self.lower_dictionary(source_class)
        self.target_class: Dict[str, str] = self.lower_dictionary(target_class)

        self.preserve_gender: bool = preserve_gender

        target_path = os.path.join(get_benchmark_output_path(), "perturbations", self.name, "chinese_name_gender.json")
        ensure_directory_exists(os.path.dirname(target_path))
        ensure_file_downloaded(source_url=self.SOURCE_URL, target_path=target_path)
        with open(os.path.join(target_path), "r", encoding="utf-8") as f:
            self.gender2name: Dict[str, List[str]] = json.load(f)
            del self.gender2name["unknown"]

            self.name2gender: Dict[str, str] = {}
            for k in self.gender2name.keys():
                for v in self.gender2name[k]:
                    self.name2gender[v] = k

    @property
    def description(self) -> PerturbationDescription:
        """Return a perturbation description for this class."""
        source_str = ",".join([f"{k}={v}" for k, v in self.source_class.items()])
        target_str = ",".join([f"{k}={v}" for k, v in self.target_class.items()])
        return ChinesePersonNamePerturbation.Description(
            name=self.name,
            fairness=True,
            prob=self.prob,
            source_class=source_str,
            target_class=target_str,
            preserve_gender=self.preserve_gender,
        )

    @staticmethod
    def lower_dictionary(d: Dict[str, str]) -> Dict[str, str]:
        """Lower the keys and values of a dictionary"""
        return dict((k.lower(), v.lower()) for k, v in d.items())

    def get_substitute_name(self, token: str, rng: Random) -> Optional[str]:
        """Get the substitute name for the token.

        Return None if self.preserve_gender tag is set, but there is no corresponding
        name in the matching gender.
        """
        options: List[str] = list(self.name2gender.keys())
        if self.preserve_gender:
            name_gender = self.name2gender[token]
            options = [n for n in self.gender2name[name_gender]]
            if not options:
                return None  # No substitution exist if we preserve the gender
            # If we don't know the gender for the source name, we randomly pick one of the target names
        name = rng.choice(list(options))
        return name

    def perturb_with_persistency(
        self, text: str, rng: Random, name_substitution_mapping: Dict[str, str], skipped_tokens: Set[str]
    ) -> str:
        """Substitute the names in text with persistency across `Instance` and their `Reference`s."""
        # Tokenize the text
        tokens, pos_tags = self.word_segment_and_pos_tagging(text)

        new_tokens: List[str] = []
        for token, tag in zip(tokens, pos_tags):
            # Find a substitution for the name, if possible
            skip: bool = token in name_substitution_mapping or token in skipped_tokens
            if not skip and token in self.name2gender:
                if rng.uniform(0, 1) < self.prob:
                    name = self.get_substitute_name(token, rng)
                    if name:
                        name_substitution_mapping[token] = name
                else:
                    skipped_tokens.add(token)

            # Substitute the token if a substitution exist
            if token in name_substitution_mapping and tag == "nr":
                token = name_substitution_mapping[token]
            new_tokens.append(token)

        return "".join(new_tokens)

    def apply(self, instance: Instance, seed: Optional[int] = None) -> Instance:
        """
        Generates a new Instance by perturbing the input, tagging the Instance and perturbing the References,
        Ensures substituted names are persistent across `Instance` and their `Reference`s.
        """
        rng: Random = self.get_rng(instance)

        # Use these to ensure that the same name replacements happen in both the instance text and the reference texts
        name_substitution_mapping: Dict[str, str] = {}
        skipped_tokens: Set[str] = set()

        references: List[Reference] = instance.references
        if self.should_perturb_references:
            references = [
                replace(
                    reference,
                    output=Output(
                        text=self.perturb_with_persistency(
                            reference.output.text, rng, name_substitution_mapping, skipped_tokens
                        )
                    ),
                    tags=reference.tags,
                )
                for reference in references
            ]

        return replace(
            instance,
            input=Input(
                text=self.perturb_with_persistency(instance.input.text, rng, name_substitution_mapping, skipped_tokens)
            ),
            references=references,
            perturbation=self.description,
        )

    @staticmethod
    def word_segment_and_pos_tagging(text: str) -> Tuple[List[str], List[str]]:
        """Perform the word segmentation and POS tagging on the text."""
        try:
            import jieba.posseg as pseg
        except ModuleNotFoundError as e:
            handle_module_not_found_error(e, ["cleva"])
        tokens: List[str] = []
        tags: List[str] = []
        output: Tuple[List[str], List[str]] = pseg.cut(text)
        for token, tag in output:
            tokens.append(token)
            tags.append(tag)

        return tokens, tags


class SimplifiedToTraditionalPerturbation(TextPerturbation):
    """Individual fairness perturbation for Chinese simplified to Chinese traditional."""

    name: str = "simplified_to_traditional"

    should_perturb_references: bool = True

    @property
    def description(self) -> PerturbationDescription:
        return PerturbationDescription(name=self.name, fairness=True)

    def __init__(
        self,
    ):
        """Initialize the Chinese simplified to Chinese traditional perturbation."""
        try:
            import opencc
        except ModuleNotFoundError as e:
            handle_module_not_found_error(e, ["cleva"])
        self.converter = opencc.OpenCC("s2t.json")

    def perturb(self, text: str, rng: Random) -> str:
        """Perform the perturbations on the provided text."""
        perturbed_text: str = self.converter.convert(text)
        return perturbed_text


class MandarinToCantonesePerturbation(TextPerturbation):
    """
    Individual fairness perturbation for Mandarin to Cantonese translation.
    The implementation is inspired by https://justyy.com/tools/chinese-converter/

    Note that this is a rule-based translation system and there are limitations.
    """

    name: str = "mandarin_to_cantonese"

    should_perturb_references: bool = True

    """ Resources """
    SOURCE_URL: str = "http://39.108.215.175/assets/conversion.json"

    @property
    def description(self) -> PerturbationDescription:
        return PerturbationDescription(name=self.name, fairness=True)

    def __init__(
        self,
    ):
        """Initialize the Mandarin to Cantonese translation perturbation."""
        try:
            import opencc
        except ModuleNotFoundError as e:
            handle_module_not_found_error(e, ["cleva"])
        self.s2t_converter = opencc.OpenCC("s2t.json")

        target_path = os.path.join(get_benchmark_output_path(), "perturbations", self.name, "conversion.json")
        ensure_directory_exists(os.path.dirname(target_path))
        ensure_file_downloaded(source_url=self.SOURCE_URL, target_path=target_path)
        with open(target_path) as fin:
            self.phrase_table = json.load(fin)

    def perturb(self, text: str, rng: Random) -> str:
        """Perform the perturbations on the provided text."""
        perturbed_text = text
        # First translate all phrases in text according to the phrase table
        for k, v in self.phrase_table.items():
            perturbed_text = perturbed_text.replace(k, v)
        # Then convert from Chinese simplified to Chinese traditional
        perturbed_text = self.s2t_converter.convert(perturbed_text)
        return perturbed_text
