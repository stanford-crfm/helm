from abc import ABC, abstractmethod
from dataclasses import replace
import itertools
from typing import List, Dict, Optional, Tuple

from proxy.models import (
    get_all_code_models,
    get_all_models,
    get_all_text_models,
    get_model_names_with_tag,
    FULL_FUNCTIONALITY_TEXT_MODEL_TAG,
    LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG,
    GPT2_TOKENIZER_TAG,
    AI21_TOKENIZER_TAG,
)
from .metric import MetricSpec
from .runner import RunSpec
from .augmentations.perturbation import PerturbationSpec
from .augmentations.data_augmenter import DataAugmenterSpec


class RunExpander(ABC):
    """
    A `RunExpander` takes a `RunSpec` and returns a list of `RunSpec`s.
    For example, it might fill out the `model` field with a variety of different models.
    """

    name: str

    @abstractmethod
    def expand(self, run_spec: RunSpec) -> List[RunSpec]:
        pass


class ReplaceValueRunExpander(RunExpander):
    """
    Replace a single field (e.g., max_train_instances) with a list of values (e.g., 0, 1, 2).
    """

    def __init__(self, value):
        """
        `value` is either the actual value to use or a lookup into the values dict.
        """
        self.name = type(self).name
        if value in type(self).values_dict:
            self.values = type(self).values_dict[value]
        else:
            self.values = [value]

    def expand(self, run_spec: RunSpec) -> List[RunSpec]:
        def sanitize(value):
            return str(value).replace("/", "_")

        return [
            replace(
                run_spec,
                name=f"{run_spec.name}{',' if ':' in run_spec.name else ':'}{self.name}={sanitize(value)}",
                adapter_spec=replace(run_spec.adapter_spec, **{self.name: value}),
            )
            for value in self.values
        ]


class ReplaceRunSpecValueRunExpander(RunExpander):
    """
    Replace a single field (e.g., max_train_instances) with a list of values (e.g., 0, 1, 2).
    """

    def __init__(self, value):
        """
        `value` is either the actual value to use or a lookup into the values dict.
        """
        self.name = type(self).name
        if value in type(self).values_dict:
            self.values = type(self).values_dict[value]
        else:
            self.values = [value]

    def expand(self, run_spec: RunSpec) -> List[RunSpec]:
        def sanitize(value):
            return str(value).replace("/", "_")

        return [
            replace(run_spec, name=f"{run_spec.name},{self.name}={sanitize(value)}", metrics=value,)
            for value in self.values
        ]


class NumTrainTrialsRunExpander(ReplaceValueRunExpander):
    """For estimating variance across runs."""

    name = "num_train_trials"
    values_dict = {"default": [3]}


class MaxTrainInstancesRunExpander(ReplaceValueRunExpander):
    """For getting learning curves."""

    name = "max_train_instances"
    values_dict = {"all": [0, 1, 2, 4, 8, 16]}


class NumOutputsRunExpander(ReplaceValueRunExpander):
    """For overriding num_outputs."""

    name = "num_outputs"
    values_dict = {"default": [1]}


DEFAULT_MODELS: List[str] = [
    "openai/davinci",
    "openai/curie",
    "openai/text-davinci-002",
    "openai/text-davinci-001",
    "openai/text-curie-001",
    "ai21/j1-jumbo",
    "ai21/j1-grande",
    "ai21/j1-large",
    "gooseai/gpt-j-6b",
    # TODO: to conserve GooseAI credits, hold off on running on GPT-NeoX until the end
    # "gooseai/gpt-neo-20b",
]


class ModelRunExpander(ReplaceValueRunExpander):
    """
    For specifying different models.
    Note: we're assuming we don't have to change the decoding parameters for different models.
    """

    name = "model"
    values_dict = {
        "full_functionality_text": get_model_names_with_tag(FULL_FUNCTIONALITY_TEXT_MODEL_TAG),
        "ai21/j1-jumbo": ["ai21/j1-jumbo"],
        "openai/curie": ["openai/curie"],
        "all": get_all_models(),
        "text": get_all_text_models(),
        "code": get_all_code_models(),
        "limited_functionality_text": get_model_names_with_tag(LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG),
        "gpt2_tokenizer": get_model_names_with_tag(GPT2_TOKENIZER_TAG),
        "ai21_tokenizer": get_model_names_with_tag(AI21_TOKENIZER_TAG),
    }


############################################################


# Helper functions to instantiate `PerturbationSpec`s.
def extra_space(num_spaces: int) -> PerturbationSpec:
    return PerturbationSpec(
        class_name="benchmark.augmentations.extra_space_perturbation.ExtraSpacePerturbation",
        args={"num_spaces": num_spaces},
    )


def space(max_spaces: int) -> PerturbationSpec:
    return PerturbationSpec(
        class_name="benchmark.augmentations.space_perturbation.SpacePerturbation", args={"max_spaces": max_spaces},
    )


def lower() -> PerturbationSpec:
    return PerturbationSpec(class_name="benchmark.augmentations.lowercase_perturbation.LowerCasePerturbation", args={})


def misspelling(prob: float) -> PerturbationSpec:
    return PerturbationSpec(
        class_name="benchmark.augmentations.misspelling_perturbation.MisspellingPerturbation", args={"prob": prob},
    )


def synonym(prob: float) -> PerturbationSpec:
    return PerturbationSpec(
        class_name="benchmark.augmentations.synonym_perturbation.SynonymPerturbation", args={"prob": prob},
    )


def contrast_sets() -> PerturbationSpec:
    return PerturbationSpec(
        class_name="benchmark.augmentations.contrast_sets_perturbation.ContrastSetsPerturbation", args={},
    )


def typo(prob: float) -> PerturbationSpec:
    return PerturbationSpec(
        class_name="benchmark.augmentations.typos_perturbation.TyposPerturbation", args={"prob": prob},
    )


def filler(prob: float) -> PerturbationSpec:
    return PerturbationSpec(
        class_name="benchmark.augmentations.filler_words_perturbation.FillerWordsPerturbation",
        args={"insert_prob": prob, "speaker_ph": False},
    )


def contract_and_expand() -> List[PerturbationSpec]:
    return [
        PerturbationSpec(
            class_name=f"benchmark.augmentations.contraction_expansion_perturbation.{mode}Perturbation", args={},
        )
        for mode in ["Contraction", "Expansion"]
    ]


def dialect(
    prob: float, source_class: str, target_class: str, mapping_file_path: Optional[str] = None
) -> PerturbationSpec:
    return PerturbationSpec(
        class_name="benchmark.augmentations.dialect_perturbation.DialectPerturbation",
        args={
            "prob": prob,
            "source_class": source_class,
            "target_class": target_class,
            "mapping_file_path": mapping_file_path,
        },
    )


def person_name(
    prob: float,
    source_class: Dict[str, str],
    target_class: Dict[str, str],
    name_file_path: Optional[str] = None,
    person_name_type: str = "first_name",
    preserve_gender: bool = True,
) -> PerturbationSpec:
    return PerturbationSpec(
        class_name="benchmark.augmentations.person_name_perturbation.PersonNamePerturbation",
        args={
            "prob": prob,
            "source_class": source_class,
            "target_class": target_class,
            "name_file_path": name_file_path,
            "person_name_type": person_name_type,
            "preserve_gender": preserve_gender,
        },
    )


def gender(
    mode: str,
    prob: float,
    source_class: str,
    target_class: str,
    mapping_file_path: Optional[str] = None,
    mapping_file_genders: Tuple[str] = None,
    bidirectional: bool = False,
) -> PerturbationSpec:
    return PerturbationSpec(
        class_name="benchmark.augmentations.gender_perturbation.GenderPerturbation",
        args={
            "mode": mode,
            "prob": prob,
            "source_class": source_class,
            "target_class": target_class,
            "mapping_file_path": mapping_file_path,
            "mapping_file_genders": mapping_file_genders,
            "bidirectional": bidirectional,
        },
    )


# Specifies the data augmentations that we're interested in trying out.
# Concretely, this is a mapping from the name (which is specified in a conf
# file or the CLI) to a list of options to try, where each option is a list of perturbations.
# Each option generates a RunSpec.
# For example, suppose:
# - We specify data_augmentation=foo
# - foo maps to {r1: [a, b], r2: [c, d, e]}
# Then we will create two RunSpecs:
# - r1: with perturbations [a, b]
# - r2: with perturbations [c, d, e]
ROBUSTNESS_PERTURBATION_SPECS: List[PerturbationSpec] = [synonym(prob=0.5), typo(prob=0.05)]

FAIRNESS_PERTURBATION_SPECS: List[PerturbationSpec] = [
    dialect(prob=1.0, source_class="SAE", target_class="AAVE"),
    gender(mode="pronouns", prob=1.0, source_class="male", target_class="female"),
    person_name(
        prob=1.0,
        source_class={"race": "white_american"},
        target_class={"race": "black_american"},
        person_name_type="first_name",
        preserve_gender=True,
    ),
]

PERTURBATION_SPECS_DICT: Dict[str, Dict[str, List[PerturbationSpec]]] = {
    # Robustness
    "extra_space": {"extra_space2": [extra_space(num_spaces=2)]},
    "contrast_sets": {"contrast_sets": [contrast_sets()]},
    "space": {"space3": [space(max_spaces=3)]},
    "lower": {"lower": [lower()]},
    "contract": {"contract": contract_and_expand()},
    "filler": {"filler0.3": [filler(0.3)]},
    "misspelling_mild": {"misspelling0.05": [misspelling(prob=0.05)]},
    "misspelling_medium": {"misspelling0.20": [misspelling(prob=0.20)]},
    "misspelling_hard": {"misspelling0.5": [misspelling(prob=0.5)]},
    "misspelling_sweep": {f"misspelling{prob}": [misspelling(prob=prob)] for prob in [0, 0.05, 0.2, 0.5]},
    "typo_easy": {"typo0.1": [typo(prob=0.10)]},
    "typo_medium": {"typo0.3": [typo(prob=0.30)]},
    "typo_hard": {"typo0.5": [typo(prob=0.50)]},
    "synonym": {"synonym0.5": [synonym(prob=0.5)]},
    # Fairness
    "dialect_easy": {
        "dialect_easy_prob=0.1_source=SAE_target=AAVE": [dialect(prob=0.1, source_class="SAE", target_class="AAVE")]
    },
    "dialect_medium": {
        "dialect_prob=0.3_source=SAE_target=AAVE": [dialect(prob=0.3, source_class="SAE", target_class="AAVE")]
    },
    "dialect_hard": {
        "dialect_prob=0.5_source=SAE_target=AAVE": [dialect(prob=0.5, source_class="SAE", target_class="AAVE")]
    },
    "dialect_deterministic": {
        "dialect_prob=1.0_source=SAE_target=AAVE": [dialect(prob=1.0, source_class="SAE", target_class="AAVE")]
    },
    "person_name_first_hard_preserve_gender": {
        "person_name_first_prob=0.5_source=white_target=black_preserve_gender=True": [
            person_name(
                prob=0.5,
                source_class={"race": "white_american"},
                target_class={"race": "black_american"},
                person_name_type="first_name",
                preserve_gender=True,
            )
        ],
    },
    "person_name_first_hard_dont_preserve_gender": {
        "person_name_first_prob=0.5_source=white_target=black_preserve_gender=False": [
            person_name(
                prob=0.5,
                source_class={"race": "white_american"},
                target_class={"race": "black_american"},
                person_name_type="first_name",
                preserve_gender=False,
            )
        ],
    },
    "person_name_first_deterministic": {
        "person_name_first_prob=1.0_source=white_target=black_preserve_gender=True": [
            person_name(
                prob=1.0,
                source_class={"race": "white_american"},
                target_class={"race": "black_american"},
                person_name_type="first_name",
                preserve_gender=True,
            )
        ],
    },
    "person_name_last_deterministic": {
        "person_name_last_prob=1.0_source=white_target=hispanic": [
            person_name(
                prob=1.0,
                source_class={"race": "white"},
                target_class={"race": "hispanic"},
                person_name_type="last_name",
                preserve_gender=False,
            )
        ],
    },
    "person_name_last_hard": {
        "person_name_last_prob=0.5_source=white_target=hispanic": [
            person_name(
                prob=0.5,
                source_class={"race": "white"},
                target_class={"race": "hispanic"},
                person_name_type="last_name",
                preserve_gender=False,
            )
        ],
    },
    "gender_terms_easy": {
        "gender_terms_prob=0.1_source=male_target=female": [
            gender(mode="terms", prob=0.1, source_class="male", target_class="female")
        ]
    },
    "gender_terms_medium": {
        "gender_terms_prob=0.3_source=male_target=female": [
            gender(mode="terms", prob=0.3, source_class="male", target_class="female")
        ]
    },
    "gender_terms_hard": {
        "gender_terms_prob=0.5_source=male_target=female": [
            gender(mode="terms", prob=0.5, source_class="male", target_class="female")
        ]
    },
    "gender_terms_deterministic": {
        "gender_terms_prob=1.0_source=male_target=female": [
            gender(mode="terms", prob=1.0, source_class="male", target_class="female")
        ]
    },
    "gender_pronouns_easy": {
        "gender_pronouns_prob=0.1_source=male_target=female": [
            gender(mode="pronouns", prob=0.1, source_class="male", target_class="female")
        ]
    },
    "gender_pronouns_medium": {
        "gender_pronouns_prob=0.3_source=male_target=female": [
            gender(mode="pronouns", prob=0.3, source_class="male", target_class="female")
        ]
    },
    "gender_pronouns_hard": {
        "gender_pronouns_prob=0.5_source=male_target=female": [
            gender(mode="pronouns", prob=0.5, source_class="male", target_class="female")
        ]
    },
    "gender_pronouns_deterministic": {
        "gender_pronouns_prob=1.0_source=male_target=female": [
            gender(mode="pronouns", prob=1.0, source_class="male", target_class="female")
        ]
    },
    "robustness": {"robustness": ROBUSTNESS_PERTURBATION_SPECS},
    "fairness": {"fairness": FAIRNESS_PERTURBATION_SPECS},
    "canonical": {"canonical": ROBUSTNESS_PERTURBATION_SPECS + FAIRNESS_PERTURBATION_SPECS},
}


class DataAugmentationRunExpander(RunExpander):
    """
    Applies a list of data augmentations, where the list of data augmentations
    is given by a name (see the keys to `PERTURBATION_SPECS_DICT` above).
    For example:
        data_augmentation=all
    Note that some names map to a single data augmentation with multiple
    perturbations (e.g., all), and others map to a list of data augmentations
    each with one perturbation (e.g., misspelling_sweep).
    """

    name = "data_augmentation"

    def __init__(self, value):
        """`value` is a comma-separated list of perturbations."""
        self.value = value

        if self.value not in PERTURBATION_SPECS_DICT:
            raise ValueError(
                f"Unknown data_augmentation: {self.value}; possible choices: {PERTURBATION_SPECS_DICT.keys()}"
            )

    def expand(self, run_spec: RunSpec) -> List[RunSpec]:
        """Return `run_spec` with data augmentations."""

        def create_run_spec(aug_name: str, perturbation_specs: List[PerturbationSpec]) -> RunSpec:
            data_augmenter_spec: DataAugmenterSpec = DataAugmenterSpec(
                perturbation_specs=perturbation_specs,
                # Always include original and perturbed instances together so that
                # we can compute the normal and robustness metrics in the same run.
                should_augment_train_instances=False,
                should_include_original_train=True,  # irrelevant
                should_skip_unchanged_train=True,  # irrelevant
                should_augment_eval_instances=True,
                should_include_original_eval=True,
                should_skip_unchanged_eval=True,
            )
            return replace(
                run_spec,
                name=f"{run_spec.name},{DataAugmentationRunExpander.name}={aug_name}",
                data_augmenter_spec=data_augmenter_spec,
            )

        return [
            create_run_spec(aug_name, perturbation_specs)
            for aug_name, perturbation_specs in PERTURBATION_SPECS_DICT[self.value].items()
        ]


RUN_EXPANDERS = dict(
    (expander.name, expander)
    for expander in [
        NumTrainTrialsRunExpander,
        MaxTrainInstancesRunExpander,
        NumOutputsRunExpander,
        ModelRunExpander,
        DataAugmentationRunExpander,
    ]
)
