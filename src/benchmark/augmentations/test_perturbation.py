from typing import List

from benchmark.scenario import Instance, Reference

from .data_augmenter import DataAugmenter
from benchmark.augmentations.perturbation import CleanPerturbation, ExtraSpacePerturbation


def test_clean_perturbation():
    instance: Instance = Instance(input="Hello my name is", references=[], tags=[])
    perturbation = CleanPerturbation()
    clean_instance: Instance = perturbation.apply("id0", instance)
    assert clean_instance.tags == ["id0", "clean"]


def test_extra_space_perturbation():
    data_augmenter = DataAugmenter(perturbations=[ExtraSpacePerturbation(num_spaces=2)], should_perturb_references=True)
    instance: Instance = Instance(
        input="Hello my name is", references=[Reference(output="some name", tags=[])], tags=[]
    )
    instances: List[Instance] = data_augmenter.generate([instance], include_original=True)
    assert len(instances) == 2
    assert instances[0].input == "Hello  my  name  is"
    assert instances[0].references[0].output == "some  name"
