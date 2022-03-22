from typing import List

from benchmark.scenario import Instance, Reference

from .data_augmenter import DataAugmenter
from .extra_space_perturbation import ExtraSpacePerturbation
from .identity_perturbation import IdentityPerturbation
from .typos_perturbation import TyposPerturbation


def test_identity_perturbation():
    instance: Instance = Instance(id="id0", input="Hello my name is", references=[])
    perturbation = IdentityPerturbation()
    clean_instance: Instance = perturbation.apply(instance)

    assert clean_instance.id == "id0"
    assert clean_instance.perturbation.name == "identity"


def test_extra_space_perturbation():
    data_augmenter = DataAugmenter(perturbations=[ExtraSpacePerturbation(num_spaces=2)], should_perturb_references=True)
    instance: Instance = Instance(
        id="id0", input="Hello my name is", references=[Reference(output="some name", tags=[])]
    )
    instances: List[Instance] = data_augmenter.generate([instance], include_original=True)

    assert len(instances) == 2
    assert instances[0].id == "id0"
    assert instances[0].perturbation.name == "extra_space"
    assert instances[0].perturbation.num_spaces == 2
    assert instances[0].input == "Hello  my  name  is"
    assert instances[0].references[0].output == "some  name"


def test_typos_perturbation():
    data_augmenter = DataAugmenter(perturbations=[TyposPerturbation(prob=0.1)], should_perturb_references=True)
    instance: Instance = Instance(
        input="After their marriage, she started a close collaboration with Karvelas.", references=[],
    )
    instances: List[Instance] = data_augmenter.generate([instance], include_original=True)

    assert len(instances) == 2
    assert instances[0].perturbation.name == "TyposPerturbation"
