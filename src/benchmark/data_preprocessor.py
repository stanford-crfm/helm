from typing import Sequence, List
from dataclasses import replace

from common.hierarchical_logger import htrack, htrack_block
from .augmentations.data_augmenter import create_data_augmenter, DataAugmenterSpec, DataAugmenter
from .scenario import Scenario, Instance, TRAIN_SPLIT, EVAL_SPLITS


class DataPreprocessor:
    """
    Gets the `Instance`s for a given `Scenario` and preprocesses them by:
    - Giving all the `Instance`s a unique ID.
    - Applying data augmentation according to `DataAugmenterSpec`.
    """

    def __init__(self, data_augmenter_spec: DataAugmenterSpec):
        self.data_augmenter_spec: DataAugmenterSpec = data_augmenter_spec

    @htrack(None)
    def preprocess(self, scenario: Scenario) -> List[Instance]:
        """
        Preprocessing steps:
        1. Gets `Instance`s for a given `Scenario`.
        2. Gives all the `Instance`s a unique ID.
        3. Applies data augmentation according to `DataAugmenterSpec`.
        """

        # Create the `Instance`s of a `Scenario`
        with htrack_block("scenario.get_instances"):
            instances: Sequence[Instance] = scenario.get_instances()

        # Give the `Instance`s of the `Scenario` a unique ID
        # Warning: Changing the id assignment logic might affect LM minimal pair evaluation (e.g. BLiMP.)
        instances = [replace(instance, id=f"id{i}") for i, instance in enumerate(instances)]

        # Create `DataAugmenter` using `DataAugmenterSpec`
        data_augmenter: DataAugmenter = create_data_augmenter(self.data_augmenter_spec)

        # Applies data augmentation to generate more train instances
        train_instances: List[Instance] = [instance for instance in instances if instance.split == TRAIN_SPLIT]
        if self.data_augmenter_spec.should_augment_train_instances:
            train_instances = data_augmenter.generate(
                train_instances,
                include_original=self.data_augmenter_spec.should_include_original_train,
                skip_unchanged=self.data_augmenter_spec.should_skip_unchanged_train,
            )

        # Applies data augmentation to generate more eval instances
        eval_instances: List[Instance] = [instance for instance in instances if instance.split in EVAL_SPLITS]
        if self.data_augmenter_spec.should_augment_eval_instances:
            eval_instances = data_augmenter.generate(
                eval_instances,
                include_original=self.data_augmenter_spec.should_include_original_eval,
                skip_unchanged=self.data_augmenter_spec.should_skip_unchanged_eval,
            )

        return train_instances + eval_instances
