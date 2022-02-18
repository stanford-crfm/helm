from typing import List

from benchmark.scenario import Instance, Reference

from .data_augmenter import DataAugmenter
from benchmark.augmentations.data_augmentation import CleanAugmentation, ExtraSpaceAugmentation


def test_clean_augmentation():
    instance: Instance = Instance(input="Hello my name is", references=[], tags=[])
    clean_data_augmentation = CleanAugmentation()
    clean_instance: Instance = clean_data_augmentation.apply("id0", instance)
    assert clean_instance.tags == ["id0", "clean"]


def test_data_augmenter():
    data_augmenter = DataAugmenter(
        data_augmentations=[ExtraSpaceAugmentation(num_spaces=2)], should_perturb_references=True
    )
    instance: Instance = Instance(
        input="Hello my name is", references=[Reference(output="some name", tags=[])], tags=[]
    )
    instances: List[Instance] = data_augmenter.generate([instance], include_original=True)
    assert instances[0].input == "Hello  my  name  is"
