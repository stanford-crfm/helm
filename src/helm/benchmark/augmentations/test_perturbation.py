# mypy: check_untyped_defs = False
from typing import List
import unittest

from helm.common.media_object import MediaObject, MultimediaObject
from helm.benchmark.scenarios.scenario import Input, Instance, Output, Reference
from helm.benchmark.augmentations.data_augmenter import DataAugmenter
from helm.benchmark.augmentations.extra_space_perturbation import ExtraSpacePerturbation
from helm.benchmark.augmentations.misspelling_perturbation import MisspellingPerturbation
from helm.benchmark.augmentations.contraction_expansion_perturbation import (
    ContractionPerturbation,
    ExpansionPerturbation,
)
from helm.benchmark.augmentations.typos_perturbation import TyposPerturbation
from helm.benchmark.augmentations.filler_words_perturbation import FillerWordsPerturbation
from helm.benchmark.augmentations.synonym_perturbation import SynonymPerturbation
from helm.benchmark.augmentations.lowercase_perturbation import LowerCasePerturbation
from helm.benchmark.augmentations.space_perturbation import SpacePerturbation
from helm.benchmark.augmentations.dialect_perturbation import DialectPerturbation
from helm.benchmark.augmentations.person_name_perturbation import PersonNamePerturbation
from helm.benchmark.augmentations.gender_perturbation import GenderPerturbation
from helm.benchmark.augmentations.suffix_perturbation import SuffixPerturbation


def test_extra_space_perturbation():
    data_augmenter = DataAugmenter(perturbations=[ExtraSpacePerturbation(num_spaces=2)])
    instance: Instance = Instance(
        id="id0", input=Input(text="Hello my name is"), references=[Reference(Output(text="some name"), tags=[])]
    )
    instances: List[Instance] = data_augmenter.generate([instance], include_original=True)

    assert len(instances) == 2
    assert instances[1].id == "id0"
    assert instances[1].perturbation.name == "extra_space"
    assert instances[1].perturbation.num_spaces == 2
    assert instances[1].input.text == "Hello  my  name  is"
    assert instances[1].references[0].output.text == "some name"


def test_multimodal_text_perturbation():
    data_augmenter = DataAugmenter(perturbations=[ExtraSpacePerturbation(num_spaces=3)])
    input: Input = Input(
        multimedia_content=MultimediaObject(
            [
                MediaObject(text="Hello what is", content_type="text/plain"),
                MediaObject(text="your name", content_type="text/plain"),
            ]
        )
    )
    instance: Instance = Instance(id="id0", input=input, references=[Reference(Output(text="some name"), tags=[])])
    instances: List[Instance] = data_augmenter.generate([instance], include_original=True)

    assert len(instances) == 2

    # Test that the first instance is unperturbed
    assert instances[0].id == "id0"
    assert instances[0].perturbation is None
    media_objects = instances[0].input.multimedia_content.media_objects
    assert media_objects[0].text == "Hello what is"
    assert media_objects[1].text == "your name"

    assert instances[1].id == "id0"
    assert instances[1].perturbation.name == "extra_space"
    media_objects = instances[1].input.multimedia_content.media_objects
    assert media_objects[0].text == "Hello   what   is"
    assert media_objects[1].text == "your   name"


def test_misspelling_perturbation():
    data_augmenter = DataAugmenter(perturbations=[MisspellingPerturbation(prob=1.0)])
    instance: Instance = Instance(
        id="id0",
        input=Input(text="Already, the new product is not available."),
        references=[],
    )
    instances: List[Instance] = data_augmenter.generate([instance], include_original=True)

    assert len(instances) == 2
    assert instances[1].id == "id0"
    assert instances[1].perturbation.name == "misspellings"
    assert instances[1].perturbation.prob == 1.0
    assert instances[1].input.text == "Alreayd, hten new product is nto avaliable."


def test_filler_words_perturbation():
    data_augmenter = DataAugmenter(perturbations=[FillerWordsPerturbation(insert_prob=0.3, speaker_ph=False)])
    instance: Instance = Instance(
        id="id0",
        input=Input(text="The quick brown fox jumps over the lazy dog."),
        references=[Reference(Output(text="The quick brown fox jumps over the lazy dog."), tags=[])],
    )
    instances: List[Instance] = data_augmenter.generate([instance], include_original=True)

    assert len(instances) == 2
    assert instances[1].id == "id0"
    assert instances[1].perturbation.name == "filler_words"
    assert instances[1].input.text == "The quick brown fox jumps over like the lazy probably dog."


def test_contraction_perturbation():
    data_augmenter = DataAugmenter(perturbations=[ContractionPerturbation()])
    instance: Instance = Instance(
        id="id0",
        input=Input(text="She is a doctor, and I am a student"),
        references=[Reference(Output(text="he is a teacher"), tags=[])],
    )
    instances: List[Instance] = data_augmenter.generate([instance], include_original=True)

    assert len(instances) == 2
    assert instances[1].id == "id0"
    assert instances[1].perturbation.name == "contraction"
    assert instances[1].input.text == "She's a doctor, and I'm a student"
    assert instances[1].references[0].output.text == "he is a teacher"


def test_expansion_perturbation():
    data_augmenter = DataAugmenter(perturbations=[ExpansionPerturbation()])
    instance: Instance = Instance(
        id="id0",
        input=Input(text="She's a doctor, and I'm a student"),
        references=[Reference(Output(text="he's a teacher"), tags=[])],
    )
    instances: List[Instance] = data_augmenter.generate([instance], include_original=True)

    assert len(instances) == 2
    assert instances[1].id == "id0"
    assert instances[1].perturbation.name == "expansion"
    assert instances[1].input.text == "She is a doctor, and I am a student"
    assert instances[1].references[0].output.text == "he's a teacher"


def test_typos_perturbation():
    data_augmenter = DataAugmenter(perturbations=[TyposPerturbation(prob=0.1)])
    instance: Instance = Instance(
        id="id0",
        input=Input(text="After their marriage, she started a close collaboration with Karvelas."),
        references=[],
    )
    instances: List[Instance] = data_augmenter.generate([instance], include_original=True)

    assert len(instances) == 2
    assert instances[1].perturbation.name == "typos"
    assert instances[1].perturbation.prob == 0.1
    assert instances[1].input.text == "After tjeir marriage, she xrwrted a cloae dollabpration with Iarvwlas."


def test_synonym_perturbation():
    data_augmenter = DataAugmenter(perturbations=[SynonymPerturbation(prob=1.0)])
    instance: Instance = Instance(
        id="id0",
        input=Input(text="This was a good movie, would watch again."),
        references=[],
    )
    instances: List[Instance] = data_augmenter.generate([instance], include_original=True)

    assert len(instances) == 2
    assert instances[1].perturbation.name == "synonym"
    assert instances[1].perturbation.prob == 1.0
    assert instances[1].input.text == "This was a near motion-picture show, would check once more."


def test_lowercase_perturbation():
    data_augmenter = DataAugmenter(perturbations=[LowerCasePerturbation()])
    instance: Instance = Instance(
        id="id0",
        input=Input(text="Hello World!\nQuite a day, huh?"),
        references=[Reference(Output(text="Yes!"), tags=[])],
    )
    instances: List[Instance] = data_augmenter.generate([instance], include_original=True)

    assert len(instances) == 2
    assert instances[1].perturbation.name == "lowercase"
    assert instances[1].input.text == "hello world!\nquite a day, huh?"
    assert instances[1].references[0].output.text == "Yes!"


def test_space_perturbation():
    data_augmenter = DataAugmenter(perturbations=[SpacePerturbation(max_spaces=3)])
    instance: Instance = Instance(id="id0", input=Input(text="Hello World!\nQuite a day, huh?"), references=[])
    instances: List[Instance] = data_augmenter.generate([instance], include_original=True)

    assert len(instances) == 2
    assert instances[1].perturbation.name == "space"
    assert instances[1].input.text == "Hello   World!\nQuite a  day,   huh?"


def test_dialect_perturbation():
    data_augmenter = DataAugmenter(
        perturbations=[DialectPerturbation(prob=1.0, source_class="SAE", target_class="AAVE")],
    )
    instance: Instance = Instance(
        id="id0",
        input=Input(text="I will remember this day to be the best day of my life."),
        references=[Reference(Output(text="Is this love?"), tags=[])],
    )
    instances: List[Instance] = data_augmenter.generate([instance], include_original=True)

    assert len(instances) == 2
    assert instances[1].perturbation.name == "dialect"
    assert instances[1].input.text == "I gon remember dis day to b the best day of mah life."
    assert instances[1].references[0].output.text == "Is dis love?"


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
    )
    instance: Instance = Instance(
        id="id0",
        input=Input(text="I learned that Jack, Peter, and Lauren are siblings! Do you know who is the oldest?"),
        references=[Reference(Output(text="Peter and peter were friends."), tags=[])],
    )
    instances: List[Instance] = data_augmenter.generate([instance], include_original=True)

    assert len(instances) == 2
    assert instances[1].perturbation.name == "person_name"
    assert (
        instances[1].input.text
        == "I learned that Lamar, Tyree, and Sharise are siblings! Do you know who is the oldest?"
    )
    assert instances[1].references[0].output.text == "Tyree and tyree were friends."


def test_gender_pronoun_perturbation():
    data_augmenter = DataAugmenter(
        perturbations=[GenderPerturbation(prob=1.0, mode="pronouns", source_class="male", target_class="female")],
    )
    instance: Instance = Instance(
        id="id0",
        input=Input(text="Did she mention that he was coming with his parents and their friends?"),
        references=[Reference(Output(text="She didn't, perhaps he didn't tell her!"), tags=[])],
    )
    instances: List[Instance] = data_augmenter.generate([instance], include_original=True)

    assert len(instances) == 2
    assert instances[1].perturbation.mode == "pronouns"
    assert instances[1].input.text == "Did she mention that she was coming with her parents and their friends?"
    assert instances[1].references[0].output.text == "She didn't, perhaps she didn't tell her!"


def test_gender_term_perturbation():
    data_augmenter = DataAugmenter(
        perturbations=[GenderPerturbation(prob=1.0, mode="terms", source_class="male", target_class="female")],
    )
    instance: Instance = Instance(
        id="id0",
        input=Input(text="His grandsons looked a lot like their dad."),
        references=[Reference(Output(text="How did their father look like?"), tags=[])],
    )
    instances: List[Instance] = data_augmenter.generate([instance], include_original=True)

    assert len(instances) == 2
    assert instances[1].perturbation.mode == "terms"
    assert instances[1].input.text == "His granddaughters looked a lot like their mom."
    assert instances[1].references[0].output.text == "How did their mother look like?"


def test_suffix_perturbation():
    data_augmenter = DataAugmenter(perturbations=[SuffixPerturbation(suffix="pixel art")])
    instance: Instance = Instance(id="id0", input=Input(text="A blue dog"), references=[])
    instances: List[Instance] = data_augmenter.generate([instance], include_original=True)

    assert len(instances) == 2
    assert instances[1].perturbation.suffix == "pixel art"
    assert instances[1].input.text == "A blue dog, pixel art"


# TODO(#1958) Fix the logic to renable this test
@unittest.skip("Currently cannot replace words at either text boundary.")
def test_gender_term_perturbation_edge_word():
    data_augmenter = DataAugmenter(
        perturbations=[GenderPerturbation(prob=1.0, mode="terms", source_class="male", target_class="female")],
    )
    instance: Instance = Instance(
        id="id0",
        input=Input(text="dad said it is okay"),
        references=[Reference(Output(text="Sure he did son"), tags=[])],
    )
    instances: List[Instance] = data_augmenter.generate([instance], include_original=False)

    assert len(instances) == 1
    assert instances[0].input.text == "mom said it is okay"
    assert instances[0].references[0].output.text == "Sure he did daughter"


# TODO(#1958) Fix the logic to renable this test
@unittest.skip("Currently cannot replace words separated by 1 character.")
def test_gender_term_perturbation_consequtive_words():
    data_augmenter = DataAugmenter(
        perturbations=[GenderPerturbation(prob=1.0, mode="terms", source_class="male", target_class="female")],
    )
    instance: Instance = Instance(
        id="id0",
        input=Input(text="I'm a dad dad: my son has a son."),
        references=[],
    )
    instances: List[Instance] = data_augmenter.generate([instance], include_original=False)

    assert len(instances) == 1
    assert instances[0].input.text == "I'm a mom mom: my daughter has a daughter."
