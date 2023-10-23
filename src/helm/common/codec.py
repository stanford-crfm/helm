"""Functions for converting to and from dataclasses."""

import dataclasses
import json
import typing
from typing import Any, Callable, Dict, List, Union, Type, TypeVar

from helm.benchmark.augmentations.cleva_perturbation import (
    ChineseTyposPerturbation,
    ChineseSynonymPerturbation,
    ChineseGenderPerturbation,
    ChinesePersonNamePerturbation,
)
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
from cattrs.gen import make_dict_structure_fn, make_dict_unstructure_fn


T = TypeVar("T")
StructureFn = Callable[[Dict[str, Any], Type[T]], T]  # dict -> dataclass
UnstructureFn = Callable[[T], Dict[str, Any]]  # dataclass -> dict


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
    # The following Perturbations are not included because
    # they use the base PerturbationDescription:
    # - ContractionPerturbation
    # - ExpansionPerturbation
    # - ContrastSetsPerturbation
    # - LowerCasePerturbation
    # - MildMixPerturbation
    ############################################################
    # CLEVA Perturbations
    ChineseTyposPerturbation.name: ChineseTyposPerturbation.Description,
    ChineseSynonymPerturbation.name: ChineseSynonymPerturbation.Description,
    ChineseGenderPerturbation.name: ChineseGenderPerturbation.Description,
    ChinesePersonNamePerturbation.name: ChinesePersonNamePerturbation.Description,
    # The following Perturbations are not included because
    # they use the base PerturbationDescription:
    # - CLEVAMildMixPerturbation
    # - SimplifiedToTraditionalPerturbation
    # - MandarinToCantonesePerturbation
}


def _build_converter() -> cattrs.Converter:
    converter = cattrs.Converter()

    # Handle omission of Nones in JSON.
    # To improve readability and reduce storage space, if a field value is None and the field
    # has no default value or a None default value, the field is omitted in the serialized JSON.
    def get_dataclass_optional_fields_without_default(cls: Type[T]) -> List[str]:
        if not dataclasses.is_dataclass(cls):
            return []
        return [
            field.name
            for field in dataclasses.fields(cls)
            if typing.get_origin(field.type) == Union and type(None) in typing.get_args(field.type)
            # For optional fields with a non-None default value, do not replace a missing value
            # with None.
            and (field.default == dataclasses.MISSING or field.default is None)
            and field.default_factory == dataclasses.MISSING
        ]

    def make_omit_nones_dict_structure_fn(cls: Type[T]) -> StructureFn[T]:
        field_names = get_dataclass_optional_fields_without_default(cls)
        _base_structure = make_dict_structure_fn(cls, converter)

        def structure(raw_dict: Dict[str, Any], inner_cls: Type[T]) -> T:
            for field_name in field_names:
                if field_name not in raw_dict:
                    raw_dict[field_name] = None
            return _base_structure(raw_dict, inner_cls)

        return structure

    def make_omit_nones_dict_unstructure_fn(cls: Type[T]) -> UnstructureFn[T]:
        field_names = get_dataclass_optional_fields_without_default(cls)
        _base_unstructure = make_dict_unstructure_fn(cls, converter)

        def structure(data: T) -> Dict[str, Any]:
            raw_dict = _base_unstructure(data)
            for field_name in field_names:
                if raw_dict[field_name] is None:
                    del raw_dict[field_name]
            return raw_dict

        return structure

    converter.register_structure_hook_factory(
        lambda cls: bool(get_dataclass_optional_fields_without_default(cls)), make_omit_nones_dict_structure_fn
    )
    converter.register_unstructure_hook_factory(
        lambda cls: bool(get_dataclass_optional_fields_without_default(cls)), make_omit_nones_dict_unstructure_fn
    )

    # Handle the use of the name field in PerturbationDescription to determine the subclass.
    base_perturbation_description_structure_fn: StructureFn = make_omit_nones_dict_structure_fn(PerturbationDescription)
    perturbation_name_to_base_structure_fn: Dict[str, StructureFn] = {
        name: make_omit_nones_dict_structure_fn(cls) for name, cls in PERTURBATION_NAME_TO_DESCRIPTION.items()
    }

    def structure_perturbation_description(
        raw_dict: Dict[Any, Any], cls: Type[PerturbationDescription]
    ) -> PerturbationDescription:
        """Convert a raw dictionary to a PerturbationDescription.
        This uses the name field to look up the correct PerturbationDescription subclass to output.
        """
        structure = perturbation_name_to_base_structure_fn.get(
            raw_dict["name"], base_perturbation_description_structure_fn
        )
        return structure(raw_dict, cls)

    converter.register_structure_hook(PerturbationDescription, structure_perturbation_description)

    return converter


_converter = _build_converter()


def from_json(data: Union[bytes, str], cls: Type[T]) -> T:
    return _converter.structure(json.loads(data), cls)


def to_json(data: Any) -> str:
    return json.dumps(_converter.unstructure(data), indent=2)


def to_json_single_line(data: Any) -> str:
    # Puts everything into a single line for readability.
    return json.dumps(_converter.unstructure(data), separators=(",", ":"))


def to_jsonl(data: List[Any]) -> str:
    return "\n".join([to_json_single_line(instance) for instance in data])


def from_jsonl(data: Union[bytes, str], cls: Type[T]) -> List[T]:
    if not isinstance(data, str):
        data = data.decode("utf-8")
    lines: List[str] = data.splitlines()
    return [from_json(line, cls) for line in lines]
