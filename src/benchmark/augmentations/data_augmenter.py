from dataclasses import dataclass, field
from typing import List

from benchmark.augmentations.data_augmentation import (
    DataAugmentation,
    DataAugmentationSpec,
    create_data_augmentation,
    CleanAugmentation,
)
from benchmark.scenario import Instance


@dataclass(frozen=True)
class DataAugmenter:

    # Data augmentations to use to generate new instances
    data_augmentations: List[DataAugmentation]

    # Whether to perturb references
    should_perturb_references: bool

    def generate(self, instances: List[Instance], include_original: bool = True) -> List[Instance]:
        """
        Given a list of Instances, generate a new list of Instances with data augmentations.
        include_original controls whether to include the original Instance when generating data augmentations.
        """
        data_augmentations = (
            self.data_augmentations + [CleanAugmentation()] if include_original else self.data_augmentations
        )

        result: List[Instance] = []
        for i, instance in enumerate(instances):
            # Tag all the augmented instances generated from the same instance with the same ID
            id_tag: str = f"id{i}"
            for data_augmentation in data_augmentations:
                result.append(data_augmentation.apply(id_tag, instance, self.should_perturb_references))
        return result


@dataclass(frozen=True)
class DataAugmenterSpec:

    # List of data augmentation specs to use to augment the data
    data_augmentation_specs: List[DataAugmentationSpec] = field(default_factory=list)

    # Whether to perturb references
    should_perturb_references: bool = False

    # Whether to augment train instances
    should_augment_train_instances: bool = False

    # Whether to include the original instances in the augmented set of train instances
    should_include_original_train: bool = False

    # Whether to augment val/test instances
    should_augment_eval_instances: bool = False

    # Whether to include the original instances in the augmented set of val/test instances
    should_include_original_eval: bool = False

    @property
    def data_augmentations(self) -> List[DataAugmentation]:
        return [create_data_augmentation(spec) for spec in self.data_augmentation_specs]

    # TODO: Get rid of this after we add the new instance fields:
    #       https://github.com/stanford-crfm/benchmarking/issues/124
    @property
    def tags(self) -> List[str]:
        tags: List[str] = [data_augmentation.tag for data_augmentation in self.data_augmentations]
        if self.should_include_original_eval:
            tags.append(CleanAugmentation.CLEAN_TAG)
        return tags


def create_data_augmenter(data_augmenter_spec: DataAugmenterSpec) -> DataAugmenter:
    """Creates a DataAugmenter from a DataAugmenterSpec."""
    return DataAugmenter(
        data_augmentations=data_augmenter_spec.data_augmentations,
        should_perturb_references=data_augmenter_spec.should_perturb_references,
    )
