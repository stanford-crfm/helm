import os
import re
from collections import defaultdict
from dataclasses import dataclass, replace
from functools import reduce
from pathlib import Path
from random import Random
from typing import Dict, List, Optional, Set

from helm.benchmark.scenarios.scenario import Input, Instance, Reference, Output
from helm.common.general import ensure_file_downloaded, ensure_directory_exists, match_case
from helm.benchmark.augmentations.perturbation_description import PerturbationDescription
from helm.benchmark.augmentations.perturbation import Perturbation
from helm.benchmark.runner import get_benchmark_output_path


# Pull this out so serialization works for multiprocessing
def lambda_defaultdict_list():
    return defaultdict(list)


class PersonNamePerturbation(Perturbation):
    """Individual fairness perturbation for person names."""

    """ Short unique identifier of the perturbation (e.g., extra_space) """
    name: str = "person_name"

    should_perturb_references: bool = True

    """ Line separator character """
    LINE_SEP = "\n"

    """ Information needed to download person_names.txt """
    FILE_NAME: str = "person_names.txt"
    SOURCE_URI: str = (
        "https://storage.googleapis.com/crfm-helm-public/source_datasets/"
        "augmentations/person_name_perturbation/person_names.txt"
    )
    OUTPUT_PATH = os.path.join(get_benchmark_output_path(), "perturbations", name)

    """ Name types """
    FIRST_NAME = "first_name"
    LAST_NAME = "last_name"
    ANY = "any"

    """ Gender categories """
    GENDER_CATEGORY = "gender"
    FEMALE = "female"
    MALE = "male"
    NEUTRAL = "neutral"
    GENDERS = [FEMALE, MALE, NEUTRAL]

    @dataclass(frozen=True)
    class Description(PerturbationDescription):
        """Description for the PersonNamePerturbation class.

        Explanation for the fields are provided in the docstring of
        PersonNamePerturbation.__init__, except source_class and target_class
        fields, which correspond to the string representation of the
        corresponding parameters passed to __init__.
        """

        prob: float = 0.0
        source_class: str = ""
        target_class: str = ""
        name_file_path: Optional[str] = None
        person_name_type: str = ""
        preserve_gender: bool = False

    def __init__(
        self,
        prob: float,
        source_class: Dict[str, str],
        target_class: Dict[str, str],
        name_file_path: Optional[str] = None,
        person_name_type: str = FIRST_NAME,
        preserve_gender: bool = True,
    ):
        """Initialize the person name perturbation.

        If name_file_path isn't provided, we use our default name mapping
        file, which can be found at:

            https://storage.googleapis.com/crfm-helm-public/source_datasets/augmentations/person_name_perturbation/person_names.txt

        The **available categories** in our default file and their values are as follows:

            If person_name_type == "last_name":

                (1) "race"   => "asian", "chinese", "hispanic", "russian", "white"

            If person_name_type == "first_name":

                (1) "race"   => "white_american", "black_american"
                (2) "gender" => "female", "male"

        The first names in our default file come from Caliskan et al. (2017),
        which derives its list from Greenwald (1998). The former removed some
        names from the latter because the corresponding tokens infrequently
        occurred in Common Crawl, which was used as the training corpus for
        GloVe. We include the full list from the latter in our default file.

        The last names in our default file and their associated categories come
        from Garg et. al. (2017), which derives its list from
        Chalabi and Flowers (2014).

        Args:
            prob: Probability of substituting a word in the source class with
                a word in the target class given that a substitution is
                available.
            source_class: The properties of the source class. The keys of the
                dictionary should correspond to categories ("race", "gender",
                "religion, "age", etc.) and the values should be the
                corresponding values. If more than one category is provided,
                the source_names list will be constructed by finding the
                intersection of the names list for the provided categories.
                Assuming the 'first_name' mode is selected, an example
                dictionary can be: {'race': 'white_american'}. Case-insensitive.
            target_class: Same as source_class, but specifies the target_class.
            name_file_path: The absolute path to a file containing the
                category associations of names. Each row of the file must
                have the following format:

                    <name>,<name_type>[,<category>,<value>]*

                Here is a breakdown of the fields:
                    <name>: The name (e.g. Alex).
                    <name_type>: Must be one of "first_name" or "last_name".
                    <category>: The name of the category (e.g. race, gender,
                        age, religion, etc.)
                    <value>: Value of the preceding category.

                [,<category>,<value>]* denotes that any number of category
                    and value pairs can be appended to a line.

                Here are some example lines:
                    li,last_name,race,chinese
                    aiesha,first_name,race,black_american,gender,female

                Notes:
                    (1) For each field, the leading and trailing spaces are
                        ignored, but those in between words in a field are
                        kept.
                    (2) All the fields are lowered.
                    (3) It is possible for a name to have multiple associations
                        (e.g. with more than one age, gender etc.)

                We use the default file if None is provided.
            person_name_type: One of "first_name" or "last_name". If
                "last_name", preserve_gender field must be False.
                Case-insensitive.
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

        person_name_type = person_name_type.lower()
        assert person_name_type in [self.FIRST_NAME, self.LAST_NAME]
        self.person_name_type = person_name_type

        self.name_file_path: Optional[str] = name_file_path
        if not self.name_file_path:
            self.name_file_path = self.download_name_file()

        # Get the possible source_names and target_names
        self.mapping_dict: Dict[str, Dict[str, List[str]]] = self.load_name_file(self.name_file_path)
        assert self.mapping_dict
        self.source_names: List[str] = self.get_possible_names(source_class)
        self.target_names: List[str] = self.get_possible_names(target_class)

        self.preserve_gender: bool = preserve_gender
        if self.preserve_gender:
            assert self.person_name_type == self.FIRST_NAME
            assert self.GENDER_CATEGORY in self.mapping_dict and len(self.mapping_dict[self.GENDER_CATEGORY])

    @property
    def description(self) -> PerturbationDescription:
        """Return a perturbation description for this class."""
        source_str = ",".join([f"{k}={v}" for k, v in self.source_class.items()])
        target_str = ",".join([f"{k}={v}" for k, v in self.target_class.items()])
        return PersonNamePerturbation.Description(
            name=self.name,
            fairness=True,
            prob=self.prob,
            source_class=source_str,
            target_class=target_str,
            name_file_path=self.name_file_path,
            person_name_type=self.person_name_type,
            preserve_gender=self.preserve_gender,
        )

    @staticmethod
    def lower_dictionary(d: Dict[str, str]) -> Dict[str, str]:
        """Lower the keys and values of a dictionary"""
        return dict((k.lower(), v.lower()) for k, v in d.items())

    def get_possible_names(self, selected_class: Dict[str, str]) -> List[str]:
        """Return possible names given a selected class, using self.mapping_dict"""
        selected_names = []
        for cat, val in selected_class.items():
            assert self.mapping_dict[cat][val]
            selected_names.append(self.mapping_dict[cat][val])
        possible_names = reduce(lambda a, b: [item for item in a if item in b], selected_names)
        return possible_names

    def download_name_file(self) -> str:
        """Download the name file from Google Cloud Storage"""
        data_path = os.path.join(self.output_path, "data")
        file_path: str = os.path.join(data_path, self.FILE_NAME)
        ensure_directory_exists(data_path)
        ensure_file_downloaded(source_url=self.SOURCE_URI, target_path=file_path)
        return file_path

    def load_name_file(self, file_path) -> Dict[str, Dict[str, List[str]]]:
        """Load the name file"""
        mapping_dict: Dict[str, Dict[str, List[str]]] = defaultdict(lambda_defaultdict_list)
        delimiter = ","
        with open(file_path, encoding="utf-8") as f:
            for line in f.readlines():
                name, name_type, *categories = line.replace(self.LINE_SEP, "").split(delimiter)
                for ind in range(len(categories) // 2):
                    category_type, category = categories[2 * ind], categories[2 * ind + 1]
                    if self.person_name_type == name_type:
                        mapping_dict[category_type][category].append(name.strip().lower())
        return mapping_dict

    def get_name_gender(self, name: str) -> Optional[str]:
        """Get the gender of the word and return one of FEMALE, MALE, and NEUTRAL."""
        gender_dict = self.mapping_dict[self.GENDER_CATEGORY]
        genders = (self.FEMALE, self.MALE, self.NEUTRAL)
        for gender in genders:
            if gender in gender_dict and name in gender_dict[gender]:
                return gender
        return None

    def get_substitute_name(self, token: str, rng: Random) -> Optional[str]:
        """Get the substitute name for the token.

        The lowered version of the token must exist in self.source_names. Return
        None if self.preserve_gender tag is set, but there is no corresponding
        name in the matching gender.
        """
        options = self.target_names
        if self.preserve_gender:
            name_gender = self.get_name_gender(token.lower())
            if name_gender:
                gendered_names_dict = self.mapping_dict[self.GENDER_CATEGORY]
                options = [n for n in self.target_names if n in gendered_names_dict[name_gender]]
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
        sep_pattern = r"([^\w])"
        tokens: List[str] = re.split(sep_pattern, text)

        new_tokens: List[str] = []
        for token in tokens:
            token_lowered: str = token.lower()

            # Find a substitution for the name, if possible
            skip: bool = token_lowered in name_substitution_mapping or token_lowered in skipped_tokens
            if not skip and token_lowered in self.source_names:
                if rng.uniform(0, 1) < self.prob:
                    name = self.get_substitute_name(token, rng)
                    if name:
                        name_substitution_mapping[token_lowered] = name
                else:
                    skipped_tokens.add(token_lowered)

            # Substitute the token if a substitution exist
            if token_lowered in name_substitution_mapping:
                substitution = name_substitution_mapping[token_lowered]
                token = match_case(token, substitution)
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
