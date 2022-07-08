from dataclasses import dataclass, field
from typing import List

from benchmark.augmentations.perturbation import (
    Perturbation,
    PerturbationSpec,
    create_perturbation,
)
from benchmark.scenario import Instance
from .identity_perturbation import IdentityPerturbation


@dataclass(frozen=True)
class DataAugmenter:

    # Perturbations to apply to generate new instances
    perturbations: List[Perturbation]

    def generate(
        self, instances: List[Instance], include_original: bool = True, skip_unchanged: bool = False
    ) -> List[Instance]:
        """
        Given a list of Instances, generate a new list of perturbed Instances.
        include_original controls whether to include the original Instance in the new list of Instances.
        skip_unchanged controls whether we include instances for which the perturbation did not change the input.
        """

        result: List[Instance] = []
        for instance in instances:
            if include_original:
                #  we want to include the original even when the perturbation does not change the input
                result.append(IdentityPerturbation().apply(instance))

            original_input: str = instance.input
            for perturbation in self.perturbations:
                perturbed_instance: Instance = perturbation.apply(instance)
                if skip_unchanged and perturbed_instance.input == original_input:
                    continue
                result.append(perturbed_instance)
        return result


@dataclass(frozen=True)
class DataAugmenterSpec:

    # List of perturbation specs to use to augment the data
    perturbation_specs: List[PerturbationSpec] = field(default_factory=list)

    # Whether to augment train instances
    should_augment_train_instances: bool = False

    # Whether to include the original instances in the augmented set of train instances
    should_include_original_train: bool = False

    # Whether to include train instances which were unaffected by the perturbation
    should_skip_unchanged_train: bool = False

    # Whether to augment val/test instances
    should_augment_eval_instances: bool = False

    # Whether to include the original instances in the augmented set of val/test instances
    should_include_original_eval: bool = False

    # Whether to include val/test instances which were unaffected by the perturbation
    should_skip_unchanged_eval: bool = False

    @property
    def perturbations(self) -> List[Perturbation]:
        return [create_perturbation(spec) for spec in self.perturbation_specs]


def create_data_augmenter(data_augmenter_spec: DataAugmenterSpec) -> DataAugmenter:
    """Creates a DataAugmenter from a DataAugmenterSpec."""
    return DataAugmenter(perturbations=data_augmenter_spec.perturbations)
