from abc import ABC, abstractmethod
from dataclasses import replace
from typing import List, Dict

from proxy.models import get_all_code_models, get_all_models, get_all_text_models
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
                name=f"{run_spec.name},{self.name}={sanitize(value)}",
                adapter_spec=replace(run_spec.adapter_spec, **{self.name: value}),
            )
            for value in self.values
        ]


class MaxTrainTrialsRunExpander(ReplaceValueRunExpander):
    """For estimating variance across runs."""

    name = "num_train_trials"
    values_dict = {"default": [5]}


class MaxTrainInstancesRunExpander(ReplaceValueRunExpander):
    """For getting learning curves."""

    name = "max_train_instances"
    values_dict = {"all": [0, 1, 2, 4, 8, 16]}


class ModelRunExpander(ReplaceValueRunExpander):
    """
    For specifying different models.
    Note: we're assuming we don't have to change the decoding parameters for different models.
    """

    name = "model"
    values_dict = {
        "default": ["openai/davinci", "ai21/j1-jumbo", "huggingface/gptj_6b"],
        "ai21/j1-jumbo": ["ai21/j1-jumbo"],
        "openai/curie": ["openai/curie"],
        "all": get_all_models(),
        "text": get_all_text_models(),
        "code": get_all_code_models(),
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
        class_name="benchmark.augmentations.misspelling_perturbation.MisspellingPerturbation", args={"prob": prob}
    )


def typo(prob: float) -> PerturbationSpec:
    return PerturbationSpec(
        class_name="benchmark.augmentations.typos_perturbation.TyposPerturbation", args={"prob": prob}
    )


def contract_and_expand() -> List[PerturbationSpec]:
    return [
        PerturbationSpec(
            class_name=f"benchmark.augmentations.contraction_expansion_perturbation.{mode}Perturbation", args={}
        )
        for mode in ["Contraction", "Expansion"]
    ]


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
PERTURBATION_SPECS_DICT: Dict[str, Dict[str, List[PerturbationSpec]]] = {
    "extra_space": {"extra_space2": [extra_space(num_spaces=2)]},
    "space": {"space3": [space(max_spaces=3)]},
    "lower": {"lower": [lower()]},
    "contract": {"contract": contract_and_expand()},
    "misspelling_mild": {"misspelling0.05": [misspelling(prob=0.05)]},
    "misspelling_medium": {"misspelling0.20": [misspelling(prob=0.20)]},
    "misspelling_hard": {"misspelling0.5": [misspelling(prob=0.5)]},
    "misspelling_sweep": {f"misspelling{prob}": [misspelling(prob=prob)] for prob in [0, 0.05, 0.2, 0.5]},
    "typo_medium": {"typo0.1": [typo(prob=0.10)]},
    "all": {"all": [misspelling(prob=0.20), space(max_spaces=3), lower(), *contract_and_expand(), typo(prob=0.1)]},
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
                should_perturb_references=False,
                # Always include original and perturbed instances together so that
                # we can compute the normal and robustness metrics in the same run.
                should_augment_train_instances=False,
                should_include_original_train=True,
                should_augment_eval_instances=True,
                should_include_original_eval=True,
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
        MaxTrainTrialsRunExpander,
        MaxTrainInstancesRunExpander,
        ModelRunExpander,
        DataAugmentationRunExpander,
    ]
)
