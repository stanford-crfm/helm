from abc import ABC, abstractmethod
from dataclasses import replace
from typing import List

from proxy.models import ALL_MODELS
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

    name = "max_train_trials"
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
        "default": ["openai/davinci"],
        "all": [model.name for model in ALL_MODELS],
        "code": ["openai/code-davinci-001", "openai/code-cushman-001"],
    }


class DataAugmentationRunExpander(RunExpander):
    """
    Applies a list of data augmentations.
    Usage:
        data_augmentation=<list of names of perturbations (e.g., typos,extra_space)>
    """

    name = "data_augmentation"

    # Mapping from short name of perturbation to the perturbation
    perturbation_specs_dict = {
        # TODO: check whether these settings are sane
        "extra_space": PerturbationSpec(
            class_name="benchmark.augmentations.extra_space_perturbation.ExtraSpacePerturbation", args={"num_spaces": 2}
        ),
        "misspelling": PerturbationSpec(
            class_name="benchmark.augmentations.misspelling_perturbation.MisspellingPerturbation", args={"prob": 0.5}
        ),
    }

    def __init__(self, value):
        """`value` is a comma-separated list of perturbations."""
        self.value = value
        self.perturbation_specs = []
        # Get the perturbations
        for name in value.split(","):
            if name == "all":
                self.perturbation_specs.extend(DataAugmentationRunExpander.perturbation_specs_dict.values())
            else:
                if name not in DataAugmentationRunExpander.perturbation_specs_dict:
                    raise ValueError(f"Unknown perturbation: {name}")
                self.perturbation_specs.append(DataAugmentationRunExpander.perturbation_specs_dict[name])

    def expand(self, run_spec: RunSpec) -> List[RunSpec]:
        """Add all the perturbations to the `run_spec`."""
        data_augmenter_spec: DataAugmenterSpec = DataAugmenterSpec(
            perturbation_specs=self.perturbation_specs,
            should_perturb_references=False,
            # TODO: are these sane defaults?
            should_augment_train_instances=True,
            should_include_original_train=True,
            should_augment_eval_instances=True,
            should_include_original_eval=True,
        )
        return [
            replace(
                run_spec,
                name=f"{run_spec.name},{DataAugmentationRunExpander.name}={self.value}",
                data_augmenter_spec=data_augmenter_spec,
            )
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
