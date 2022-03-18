import random
from typing import List

from benchmark.augmentations.perturbation import (
    IdentityPerturbation,
    ExtraSpacePerturbation,
    CityNameReplacementPerturbation,
)
from benchmark.scenario import Instance, Reference
from .data_augmenter import DataAugmenter


def test_identity_perturbation():
    instance: Instance = Instance(input="Hello my name is", references=[])
    perturbation = IdentityPerturbation()
    clean_instance: Instance = perturbation.apply("id0", instance)

    assert clean_instance.id == "id0"
    assert clean_instance.perturbation.name == "identity"


def test_extra_space_perturbation():
    data_augmenter = DataAugmenter(perturbations=[ExtraSpacePerturbation(num_spaces=2)], should_perturb_references=True)
    instance: Instance = Instance(input="Hello my name is", references=[Reference(output="some name", tags=[])])
    instances: List[Instance] = data_augmenter.generate([instance], include_original=True)

    assert len(instances) == 2
    assert instances[0].id == "id0"
    assert instances[0].perturbation.name == "extra_space"
    assert instances[0].perturbation.num_spaces == 2
    assert instances[0].input == "Hello  my  name  is"
    assert instances[0].references[0].output == "some  name"


def test_city_name_replacement_perturbation():
    def _perturb(text) -> str:
        instance = Instance(input=text, references=[])
        random.seed(123)
        perturbation = CityNameReplacementPerturbation()
        output: Instance = perturbation.apply("id0", instance)
        assert output.id == "id0"
        assert output.perturbation.name == "city_name_replacement"  # type: ignore
        return output.input

    output = _perturb("I think San Francisco is a nice city.")
    assert output == "I think Bell is a nice city."

    # Not in the populous city list (but in the scarcely populated city list)
    text = "I think Stanford is a nice place."
    output = _perturb(text)
    assert output == text

    # Starts with a city name
    output = _perturb("San Francisco is a nice city.")
    assert output == "Bell is a nice city."

    # Non-city mentions
    output = _perturb("San Francisco Giants is looking at buying properties in San Francisco.")
    assert output == "San Francisco Giants is looking at buying properties in Bell."

    # Multiple mentions
    output = _perturb(
        "San Jose is a nice city near San Francisco. It is one hour drive from San Jose to San Francisco."
    )
    assert output == "Forest Park is a nice city near Bell. It is one hour drive from Forest Park to Bell."
