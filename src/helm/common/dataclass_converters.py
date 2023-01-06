"""Functions for converting to and from dataclasses."""

import dataclasses
import typing
from typing import Any, Callable, Dict, List, Union, Type, TypeVar

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

import cattrs
from cattrs.gen import make_dict_structure_fn


T = TypeVar("T")


# TODO(#1251): Add proper class registration
PERTURBATION_NAME_TO_DESCRIPTION = {
    DialectPerturbation.name: DialectPerturbation.Description,
    ExtraSpacePerturbation.name: ExtraSpacePerturbation.Description,
    FillerWordsPerturbation.name: FillerWordsPerturbation.Description,
    GenderPerturbation.name: GenderPerturbation.Description,
    MisspellingPerturbation.name: MisspellingPerturbation.Description,
    PersonNamePerturbation.name: PersonNamePerturbation.Description,
    SpacePerturbation.name: SpacePerturbation.Description,
    SynonymPerturbation.name: SynonymPerturbation.Description,
    TyposPerturbation.name: TyposPerturbation.Description,
}


def _build_converter() -> cattrs.Converter:
    converter = cattrs.Converter()
    # Handle the use of the name field in PerturbationDescription to determine the subclass.
    base_structure_perturbation_description = make_dict_structure_fn(PerturbationDescription, converter)

    def _structure_perturbation_description(
        raw_dict: Dict[Any, Any], cls: Type[PerturbationDescription]
    ) -> PerturbationDescription:
        """Convert a raw dictionary to a PerturbationDescription.
        This uses the name field to look up the correct PerturbationDescription subclass to output.
        """
        subclass = PERTURBATION_NAME_TO_DESCRIPTION.get(raw_dict["name"])
        if subclass is None:
            return base_structure_perturbation_description(raw_dict, cls)
        return converter.structure(raw_dict, subclass)

    converter.register_structure_hook(lambda cls: cls == PerturbationDescription, _structure_perturbation_description)

    # Handle omission of Nones in JSON.
    def get_dataclass_optional_fields_without_default(cls: Type) -> List[str]:
        if not dataclasses.is_dataclass(cls):
            return []
        return [
            field.name
            for field in dataclasses.fields(cls)
            if typing.get_origin(field.type) == Union
            and type(None) in typing.get_args(field.type)
            and field.default == dataclasses.MISSING
            and field.default == dataclasses.MISSING
        ]

    def has_dataclass_optional_fields_without_default(cls: Type) -> bool:
        return bool(get_dataclass_optional_fields_without_default(cls))

    def make_omit_nones_dict_structure_fn(cls: Type) -> Callable[[Dict[Any, Any], Type[T]], T]:
        field_names = get_dataclass_optional_fields_without_default(cls)
        _base_structure = make_dict_structure_fn(cls, converter)

        def structure(raw_dict: Dict[Any, Any], inner_cls: Type[T]) -> T:
            for field_name in field_names:
                if field_name not in raw_dict:
                    raw_dict[field_name] = None
            return _base_structure(raw_dict, inner_cls)

        return structure

    converter.register_structure_hook_factory(
        has_dataclass_optional_fields_without_default, make_omit_nones_dict_structure_fn
    )

    return converter


_converter = _build_converter()


def structure(raw: Any, cls: Type[T]) -> T:
    return _converter.structure(raw, cls)
