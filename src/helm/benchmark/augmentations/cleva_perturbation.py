from dataclasses import dataclass
import json
import os
from random import Random
from typing import Dict, List, Optional

import unidecode
import pypinyin
import jieba

from helm.common.general import ensure_file_downloaded, ensure_directory_exists
from .perturbation_description import PerturbationDescription
from .perturbation import Perturbation


############################ Robustness ################################


class ButterFingerPerturbation(Perturbation):
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
        rare_word_prob: Optional[float] = 0.05
        consider_tone: Optional[bool] = False
        word_level_perturb: Optional[bool] = True

    name: str = "butter_finger"

    # For downloading resources
    ASSET_URL = "http://emnlp.clevaplat.com:8001/assets/butter_finger"
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
        rare_char_prob: Optional[float] = 0.05,
        consider_tone: Optional[bool] = False,
        word_level_perturb: Optional[bool] = True,
    ):
        # Assign parameters to instance variables
        self.prob: float = prob
        self.rare_char_prob: float = rare_char_prob  # How likely we will use rare Chinese characters
        self.consider_tone: bool = (
            consider_tone  # Should we take the tone of Pinyin into account when considering similar char/words
        )
        self.word_level_perturb: bool = word_level_perturb  # Whether we perturb text on the character or word level

        # Ensure all necessary data are downloaded
        output_dir = os.path.join("benchmark_output", "perturbations", self.name)
        ensure_directory_exists(output_dir)
        for FILE_NAME in self.FILE_NAMES:
            target_path = os.path.join(output_dir, FILE_NAME)
            SOURCE_URI: str = f"{self.ASSET_URL}/{FILE_NAME}"
            ensure_file_downloaded(source_url=SOURCE_URI, target_path=target_path)

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
        return ButterFingerPerturbation.Description(name=self.name, robustness=True, prob=self.prob)

    def perturb(self, text: str, rng: Random) -> str:
        butter_text: str = ""
        output: List[str] = jieba.lcut(text)
        if self.word_level_perturb:
            words_and_similar_word_dict_list = self.get_words_with_similar_pinyin(
                output,
                self.rare_char_prob,
                self.chinese_character_database,
                self.common_chinese_character_database,
                self.chinese_words_database,
                self.consider_tone,
                rng,
            )
            for dict in words_and_similar_word_dict_list:
                original_word = dict["original_word"]
                similar_pinyins_words = dict["similar_pinyin_words"]
                if rng.random() <= self.prob and len(similar_pinyins_words) != 0:
                    new_chinese_character = rng.choice(similar_pinyins_words)
                else:
                    new_chinese_character = original_word
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
        chinese_character,
        rare_word_prob,
        chinese_character_database,
        common_chinese_character_database,
        consider_tone,
        rng,
    ):

        pinyin_for_char_to_be_perturbed = pypinyin.pinyin(chinese_character)
        pinyin_for_char_to_be_perturbed = [item for pinyin in pinyin_for_char_to_be_perturbed for item in pinyin]
        pinyin_for_char_to_be_perturbed = "".join(pinyin_for_char_to_be_perturbed)

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
        text,
        rare_word_prob,
        chinese_character_database,
        common_chinese_character_database,
        chinese_words_database,
        consider_tone,
        rng,
    ):
        words_and_similar_word_dict_list = []
        for original_word in text:
            words_and_similar_word_dict = {"original_word": original_word}
            original_word_len = len(original_word)
            similar_word_pinyin_list = []
            similar_word_pinyin_list = self.get_similar_word_pinyin_list(
                chinese_character_database,
                chinese_words_database,
                common_chinese_character_database,
                consider_tone,
                original_word,
                original_word_len,
                rare_word_prob,
                similar_word_pinyin_list,
                rng,
            )

            words_and_similar_word_dict["similar_pinyin_words"] = similar_word_pinyin_list
            words_and_similar_word_dict_list.append(words_and_similar_word_dict)
        return words_and_similar_word_dict_list

    def get_similar_word_pinyin_list(
        self,
        chinese_character_database,
        chinese_words_database,
        common_chinese_character_database,
        consider_tone,
        original_word,
        original_word_len,
        rare_word_prob,
        similar_word_pinyin_list,
        rng,
    ):
        if original_word_len == 1:
            similar_pinyins = self.get_characters_with_similar_pinyin(
                original_word,
                rare_word_prob,
                chinese_character_database,
                common_chinese_character_database,
                consider_tone,
                rng,
            )
            similar_word_pinyin_list = [char for char in similar_pinyins]
        elif original_word_len > 1:
            original_word_pinyins = pypinyin.pinyin(original_word)
            original_word_pinyins_flatten = [item for pinyin in original_word_pinyins for item in pinyin]
            original_word_pinyins_string = "".join(original_word_pinyins_flatten)
            if not consider_tone:
                original_word_pinyins_string = unidecode.unidecode(original_word_pinyins_string)
            candidate_words = chinese_words_database.get(original_word_pinyins_string, [])
            for word in candidate_words:
                if word != original_word:
                    similar_word_pinyin_list.append(word)
        return similar_word_pinyin_list

    def retrieve_from_database(
        self,
        chinese_character,
        chars_with_similar_pinyin,
        chinese_character_database,
        consider_tone,
        pinyin_for_char_to_be_perturbed,
    ):
        if not consider_tone:
            pinyin_for_char_to_be_perturbed = unidecode.unidecode(pinyin_for_char_to_be_perturbed)
        candidate_chars = chinese_character_database.get(pinyin_for_char_to_be_perturbed, [])
        for char in candidate_chars:
            if chinese_character != char:
                chars_with_similar_pinyin += char
        return chars_with_similar_pinyin


class ChineseSynonymPerturbation(Perturbation):
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
        trial_num: Optional[int] = 10

    name: str = "chinese_synonym"

    # For downloading resources
    SOURCE_URI: str = "http://emnlp.clevaplat.com:8001/assets/synonyms.json"

    def __init__(self, prob: float, trial_num: Optional[int] = 10):
        # Assign parameters to instance variables
        self.prob: float = prob
        self.trial_num: int = trial_num  # Number of trial to get a 100% perturbed text

        target_dir = os.path.join("benchmark_output", "perturbations", self.name, "synonyms.json")
        ensure_directory_exists(os.path.dirname(target_dir))
        ensure_file_downloaded(source_url=self.SOURCE_URI, target_path=target_dir)
        with open(os.path.join(target_dir)) as f:
            self.synonym_dict: Dict[str, List[str]] = json.load(f)

    @property
    def description(self) -> PerturbationDescription:
        return ChineseSynonymPerturbation.Description(name=self.name, robustness=True, prob=self.prob)

    def perturb(self, text: str, rng: Random) -> str:
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

    def sample_word(self, sample_list: list, rng: Random):
        index = rng.randint(0, len(sample_list) - 1)
        return sample_list[index]


class CLEVAMildMixPerturbation(Perturbation):
    """
    CLEVA robustness perturbation that composes several perturbations.
    """

    name: str = "cleva_mild_mix"

    # Don't perturb references because it's not fair to have to generate broken text.
    should_perturb_references: bool = False

    def __init__(self):
        self.synonym_perturbation = ChineseSynonymPerturbation(0.3)
        self.butter_finger_perturbation = ButterFingerPerturbation(0.05)

    @property
    def description(self) -> PerturbationDescription:
        return PerturbationDescription(name=self.name, robustness=True)

    def perturb(self, text: str, rng: Random) -> str:
        # Original CLEVA paper also adopts the "character swapping",
        # but we find that it has a negative impact on many reasoning
        # tasks. Therefore we do not include it here.
        text = self.synonym_perturbation.perturb(text, rng)
        text = self.butter_finger_perturbation.perturb(text, rng)
        return text


############################ Fairness ################################
