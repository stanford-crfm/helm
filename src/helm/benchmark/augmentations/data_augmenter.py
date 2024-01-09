from dataclasses import dataclass, field
from typing import List

from helm.common.hierarchical_logger import htrack, hlog
from helm.common.general import parallel_map
from helm.benchmark.augmentations.perturbation import (
    Perturbation,
    PerturbationSpec,
    create_perturbation,
)
from helm.benchmark.scenarios.scenario import Instance


@dataclass(frozen=True)
class Processor:
    include_original: bool
    skip_unchanged: bool
    perturbations: List[Perturbation]
    seeds_per_instance: int

    def process(self, instance: Instance) -> List[Instance]:
        result: List[Instance] = []
        if self.include_original:
            #  we want to include the original even when the perturbation does not change the input
            result.append(instance)

        for perturbation in self.perturbations:
            for i in range(self.seeds_per_instance):
                perturbed_instance: Instance = perturbation.apply(instance, seed=None if i == 0 else i)
                if self.skip_unchanged and perturbed_instance.input == instance.input:
                    continue
                result.append(perturbed_instance)
        return result


@dataclass(frozen=True)
class DataAugmenter:
    # Perturbations to apply to generate new instances
    perturbations: List[Perturbation]

    @htrack(None)
    def generate(
        self,
        instances: List[Instance],
        include_original: bool = True,
        skip_unchanged: bool = False,
        seeds_per_instance: int = 1,
        parallelism: int = 1,
    ) -> List[Instance]:
        """
        Given a list of Instances, generate a new list of perturbed Instances.
        include_original controls whether to include the original Instance in the new list of Instances.
        skip_unchanged controls whether we include instances for which the perturbation did not change the input.
        """
        processor = Processor(
            include_original=include_original,
            skip_unchanged=skip_unchanged,
            perturbations=self.perturbations,
            seeds_per_instance=seeds_per_instance,
        )
        results: List[List[Instance]] = parallel_map(
            processor.process,
            instances,
            parallelism=parallelism,
        )
        output_instances = [instance for result in results for instance in result]

        hlog(f"{len(instances)} instances augmented to {len(output_instances)} instances")
        return output_instances


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

    # How many different seeds to apply each perturbation with
    seeds_per_instance: int = 1

    @property
    def perturbations(self) -> List[Perturbation]:
        return [create_perturbation(spec) for spec in self.perturbation_specs]


def create_data_augmenter(data_augmenter_spec: DataAugmenterSpec) -> DataAugmenter:
    """Creates a DataAugmenter from a DataAugmenterSpec."""
    return DataAugmenter(perturbations=data_augmenter_spec.perturbations)
