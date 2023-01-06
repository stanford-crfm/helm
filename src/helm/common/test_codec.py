import unittest
import json

from dataclasses import dataclass

from typing import Dict, List, Optional

from helm.benchmark.augmentations.dialect_perturbation import DialectPerturbation
from helm.benchmark.augmentations.extra_space_perturbation import ExtraSpacePerturbation
from helm.benchmark.augmentations.filler_words_perturbation import FillerWordsPerturbation
from helm.benchmark.augmentations.gender_perturbation import GenderPerturbation
from helm.benchmark.augmentations.misspelling_perturbation import MisspellingPerturbation
from helm.benchmark.augmentations.person_name_perturbation import PersonNamePerturbation
from helm.benchmark.augmentations.space_perturbation import SpacePerturbation
from helm.benchmark.augmentations.synonym_perturbation import SynonymPerturbation
from helm.benchmark.augmentations.typos_perturbation import TyposPerturbation
from helm.benchmark.augmentations.perturbation_description import PerturbationDescription
from helm.common.codec import from_json, to_json


@dataclass(frozen=True)
class DataClassChildForTest:
    required_int: int


@dataclass(frozen=True)
class DataClassWithOptionals:
    optional_str: Optional[str]
    optional_int: Optional[int]
    optional_bool: Optional[bool]
    optional_list: Optional[List[int]]
    optional_dict: Optional[Dict[str, int]]
    optional_child: Optional[DataClassChildForTest]


@dataclass(frozen=True)
class DataClassWithDefaults:
    required_int_with_default: int = -1
    optional_int_with_int_default: Optional[int] = -2
    optional_int_with_none_default: Optional[int] = None


class TestJsonCodec(unittest.TestCase):
    def test_round_trip_optional(self):
        data = DataClassWithOptionals(
            optional_str="hello",
            optional_int=42,
            optional_bool=True,
            optional_list=[2, 3, 5],
            optional_dict={"x": 7},
            optional_child=DataClassChildForTest(137),
        )
        self.assertEqual(data, from_json(to_json(data), DataClassWithOptionals))

    def test_round_trip_optional_nones(self):
        data = DataClassWithOptionals(
            optional_str=None,
            optional_int=None,
            optional_bool=None,
            optional_list=None,
            optional_dict=None,
            optional_child=None,
        )
        data_json = to_json(data)
        self.assertEqual("{}", data_json)
        self.assertEqual(data, from_json(data_json, DataClassWithOptionals))

    def test_round_trip_default(self):
        data = DataClassWithDefaults()
        data_json = to_json(data)
        self.assertCountEqual(
            {"required_int_with_default": -1, "optional_int_with_int_default": -2}.items(),
            json.loads(data_json).items(),
        )
        self.assertEqual(data, from_json(data_json, DataClassWithDefaults))

    def test_round_trip_default_ints(self):
        data = DataClassWithDefaults(
            required_int_with_default=1,
            optional_int_with_int_default=2,
            optional_int_with_none_default=3,
        )
        data_json = to_json(data)
        self.assertEqual(data, from_json(data_json, DataClassWithDefaults))

    def test_round_trip_default_nones(self):
        data = DataClassWithDefaults(
            optional_int_with_int_default=None,
            optional_int_with_none_default=None,
        )
        data_json = to_json(data)
        self.assertCountEqual(
            {
                "required_int_with_default": -1,
                # `optional_int_with_int_default` should deserialize back to None,
                # rather than the default int value. Therefore it must be
                # serialized to null in JSON instead of removed.
                "optional_int_with_int_default": None,
            }.items(),
            json.loads(data_json).items(),
        )
        self.assertEqual(data, from_json(data_json, DataClassWithDefaults))

    def test_round_trip_perturbation_descriptions(self):
        descriptions = [
            PerturbationDescription(
                name="unknown",
            ),
            DialectPerturbation.Description(
                name=DialectPerturbation.name,
                fairness=True,
                prob=0.5,
                source_class="source_class",
                target_class="target_class",
                mapping_file_path="mapping_file_path",
            ),
            ExtraSpacePerturbation.Description(name=ExtraSpacePerturbation.name, robustness=True, num_spaces=2),
            FillerWordsPerturbation.Description(name=FillerWordsPerturbation.name, robustness=True, insert_prob=0.5),
            GenderPerturbation.Description(
                name=GenderPerturbation.name,
                mode="mode",
                fairness=True,
                prob=0.5,
                source_class="source_class",
                target_class="target_class",
                bidirectional=True,
            ),
            MisspellingPerturbation.Description(name=MisspellingPerturbation.name, robustness=True, prob=0.5),
            PersonNamePerturbation.Description(
                name=PersonNamePerturbation.name,
                fairness=True,
                prob=0.5,
                source_class="source_str",
                target_class="target_str",
                name_file_path="name_file_path",
                person_name_type="person_name_type",
                preserve_gender=True,
            ),
            SpacePerturbation.Description(name=SpacePerturbation.name, robustness=True, max_spaces=2),
            SynonymPerturbation.Description(name=SynonymPerturbation.name, robustness=True, prob=0.5),
            TyposPerturbation.Description(name=TyposPerturbation.name, robustness=True, prob=0.5),
        ]
        for description in descriptions:
            self.assertEqual(description, from_json(to_json(description), PerturbationDescription))
