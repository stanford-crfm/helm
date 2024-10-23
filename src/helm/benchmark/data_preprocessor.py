from typing import List
from helm.common.hierarchical_logger import htrack
from helm.benchmark.augmentations.data_augmenter import create_data_augmenter, DataAugmenterSpec, DataAugmenter
from helm.benchmark.scenarios.scenario import Instance, TRAIN_SPLIT, EVAL_SPLITS


class DataPreprocessor:
    """
    Applying data augmentation according to `DataAugmenterSpec`.
    """

    def __init__(self, data_augmenter_spec: DataAugmenterSpec):
        self.data_augmenter_spec: DataAugmenterSpec = data_augmenter_spec

    @htrack(None)
    def preprocess(self, instances: List[Instance], parallelism: int = 1) -> List[Instance]:
        """
        Applies data augmentation according to `DataAugmenterSpec`.
        """
        # Create `DataAugmenter` using `DataAugmenterSpec`
        data_augmenter: DataAugmenter = create_data_augmenter(self.data_augmenter_spec)

        # Applies data augmentation to generate more train instances
        train_instances: List[Instance] = [instance for instance in instances if instance.split == TRAIN_SPLIT]
        if self.data_augmenter_spec.should_augment_train_instances:
            train_instances = data_augmenter.generate(
                train_instances,
                include_original=self.data_augmenter_spec.should_include_original_train,
                skip_unchanged=self.data_augmenter_spec.should_skip_unchanged_train,
                seeds_per_instance=self.data_augmenter_spec.seeds_per_instance,
                parallelism=parallelism,
            )

        # Applies data augmentation to generate more eval instances
        eval_instances: List[Instance] = [instance for instance in instances if instance.split in EVAL_SPLITS]
        if self.data_augmenter_spec.should_augment_eval_instances:
            eval_instances = data_augmenter.generate(
                eval_instances,
                include_original=self.data_augmenter_spec.should_include_original_eval,
                skip_unchanged=self.data_augmenter_spec.should_skip_unchanged_eval,
                seeds_per_instance=self.data_augmenter_spec.seeds_per_instance,
                parallelism=parallelism,
            )

        return train_instances + eval_instances
