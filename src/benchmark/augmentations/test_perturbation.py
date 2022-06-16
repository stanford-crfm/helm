from typing import List

from benchmark.scenario import Instance, Reference

from .data_augmenter import DataAugmenter
from .extra_space_perturbation import ExtraSpacePerturbation
from .identity_perturbation import IdentityPerturbation
from .misspelling_perturbation import MisspellingPerturbation
from .contraction_expansion_perturbation import ContractionPerturbation, ExpansionPerturbation
from .typos_perturbation import TyposPerturbation
from .filler_words_perturbation import FillerWordsPerturbation
from .synonym_perturbation import SynonymPerturbation
from .lowercase_perturbation import LowerCasePerturbation
from .space_perturbation import SpacePerturbation
from .dialect_perturbation import DialectPerturbation
from .person_name_perturbation import PersonNamePerturbation
from .gender_perturbation import GenderPerturbation


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


def test_misspelling_perturbation():
    data_augmenter = DataAugmenter(perturbations=[MisspellingPerturbation(prob=1.0)], should_perturb_references=True)
    instance: Instance = Instance(
        id="id0", input="Already, the new product is not available.", references=[],
    )
    instances: List[Instance] = data_augmenter.generate([instance], include_original=True)

    assert len(instances) == 2
    assert instances[0].id == "id0"
    assert instances[0].perturbation.name == "misspellings"
    assert instances[0].perturbation.prob == 1.0
    assert instances[0].input == "Alreayd, teh new product is nto availaible."


def test_filler_words_perturbation():
    data_augmenter = DataAugmenter(
        perturbations=[FillerWordsPerturbation(insert_prob=0.3, speaker_ph=False)], should_perturb_references=True
    )
    instance: Instance = Instance(
        id="id0",
        input="The quick brown fox jumps over the lazy dog.",
        references=[Reference(output="The quick brown fox jumps over the lazy dog.", tags=[])],
    )
    instances: List[Instance] = data_augmenter.generate([instance], include_original=True)

    assert len(instances) == 2
    assert instances[0].id == "id0"
    assert instances[0].perturbation.name == "filler_words"
    assert instances[0].input == "The probably quick like brown like fox err jumps over the lazy dog."


def test_contraction_perturbation():
    data_augmenter = DataAugmenter(perturbations=[ContractionPerturbation()], should_perturb_references=True)
    instance: Instance = Instance(
        id="id0", input="She is a doctor, and I am a student", references=[Reference(output="he is a teacher", tags=[])]
    )
    instances: List[Instance] = data_augmenter.generate([instance], include_original=True)

    assert len(instances) == 2
    assert instances[0].id == "id0"
    assert instances[0].perturbation.name == "contraction"
    assert instances[0].input == "She's a doctor, and I'm a student"
    assert instances[0].references[0].output == "he's a teacher"


def test_expansion_perturbation():
    data_augmenter = DataAugmenter(perturbations=[ExpansionPerturbation()], should_perturb_references=True)
    instance: Instance = Instance(
        id="id0", input="She's a doctor, and I'm a student", references=[Reference(output="he's a teacher", tags=[])]
    )
    instances: List[Instance] = data_augmenter.generate([instance], include_original=True)

    assert len(instances) == 2
    assert instances[0].id == "id0"
    assert instances[0].perturbation.name == "expansion"
    assert instances[0].input == "She is a doctor, and I am a student"
    assert instances[0].references[0].output == "he is a teacher"


def test_typos_perturbation():
    data_augmenter = DataAugmenter(perturbations=[TyposPerturbation(prob=0.1)], should_perturb_references=True)
    instance: Instance = Instance(
        id="id0", input="After their marriage, she started a close collaboration with Karvelas.", references=[],
    )
    instances: List[Instance] = data_augmenter.generate([instance], include_original=True)

    assert len(instances) == 2
    assert instances[0].perturbation.name == "TyposPerturbation"
    assert instances[0].perturbation.prob == 0.1
    assert instances[0].input == "After their marrjage, she started a close collaborwtion with Karvelas."


def test_synonym_perturbation():
    data_augmenter = DataAugmenter(perturbations=[SynonymPerturbation(prob=1.0)], should_perturb_references=True)
    instance: Instance = Instance(
        id="id0", input="This was a good movie, would watch again.", references=[],
    )
    instances: List[Instance] = data_augmenter.generate([instance], include_original=True)

    assert len(instances) == 2
    assert instances[0].perturbation.name == "SynonymPerturbation"
    assert instances[0].perturbation.prob == 1.0
    assert instances[0].input == "This was a dependable movie, would determine again."


def test_lowercase_perturbation():
    data_augmenter = DataAugmenter(perturbations=[LowerCasePerturbation()], should_perturb_references=True)
    instance: Instance = Instance(
        id="id0", input="Hello World!\nQuite a day, huh?", references=[Reference(output="Yes!", tags=[])],
    )
    instances: List[Instance] = data_augmenter.generate([instance], include_original=True)

    assert len(instances) == 2
    assert instances[0].perturbation.name == "lowercase"
    assert instances[0].input == "hello world!\nquite a day, huh?"
    assert instances[0].references[0].output == "yes!"


def test_space_perturbation():
    data_augmenter = DataAugmenter(perturbations=[SpacePerturbation(max_spaces=3)], should_perturb_references=False)
    instance: Instance = Instance(id="id0", input="Hello World!\nQuite a day, huh?", references=[])
    instances: List[Instance] = data_augmenter.generate([instance], include_original=True)

    print(instances)
    assert len(instances) == 2
    assert instances[0].perturbation.name == "space"
    assert instances[0].input == "Hello   World!\nQuite   aday,  huh?"


def test_dialect_perturbation():
    data_augmenter = DataAugmenter(
        perturbations=[DialectPerturbation(prob=1.0, source_class="SAE", target_class="AAVE")],
        should_perturb_references=True,
    )
    instance: Instance = Instance(
        id="id0",
        input="I will remember this day to be the best day of my life.",
        references=[Reference(output="Is this love?", tags=[])],
    )
    instances: List[Instance] = data_augmenter.generate([instance], include_original=True)

    print(instances)
    assert len(instances) == 2
    assert instances[0].perturbation.name == "dialect"
    assert instances[0].input == "I gon remember dis day to b the best day of mah life."
    assert instances[0].references[0].output == "Is dis love?"


def test_person_name_perturbation():
    data_augmenter = DataAugmenter(
        perturbations=[
            PersonNamePerturbation(
                prob=1.0,
                source_class={"race": "white_american"},
                target_class={"race": "black_american"},
                person_name_type="first_name",
                preserve_gender=True,
            )
        ],
        should_perturb_references=True,
    )
    instance: Instance = Instance(
        id="id0",
        input="I learned that Jack, Peter, and Lauren are siblings! Do you know who is the oldest?",
        references=[Reference(output="Peter", tags=[])],
    )
    instances: List[Instance] = data_augmenter.generate([instance],
                                                        include_original=True)

    print(instances)
    assert len(instances) == 2
    assert instances[0].perturbation.name == "person_name"
    assert (
            instances[
                0].input == "I learned that Jamel, Wardell, and Latoya are siblings! Do you know who is the oldest?"
    )
    assert instances[0].references[0].output == "Wardell"
