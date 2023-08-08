"""This allows users to register additional models via configuration file.

TODO(#1673): Support adding other kinds of models besides AI21.
"""

from typing import Any, Dict, Iterable, Optional, List
from dataclasses import dataclass, field
from datetime import date

import cattrs
import dacite
import yaml
from helm.benchmark.presentation.schema import ModelField

from helm.common.general import parse_hocon
from helm.proxy.models import ALL_MODELS, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, MODEL_NAME_TO_MODEL, TEXT_MODEL_TAG, Model


@dataclass(frozen=True)
class ModelMetadata:
    name: str

    # Organization that originally created the model (e.g. "EleutherAI")
    #   Note that this may be different from group or the prefix of the model `name`
    #   ("together" in "together/gpt-j-6b") as the hosting organization
    #   may be different from the creator organization. We also capitalize
    #   this field properly to later display in the UI.
    # TODO: in the future, we want to cleanup the naming in the following ways:
    # - make the creator_organization an identifier with a separate display name
    # - have a convention like <hosting_organization><creator_organization>/<model_name>
    creator_organization: Optional[str] = None

    # How this model is available (e.g., limited)
    access: Optional[str] = None

    # Whether we have yet to evaluate this model
    todo: bool = False

    # When was the model released
    release_date: Optional[date] = None

    # The number of parameters
    # This should be a string as the number of parameters is usually a round number (175B),
    # but we set it as an int for plotting purposes.
    num_parameters: Optional[int] = None

    # Tags corresponding to the properties of the model.
    tags: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class ModelMetadataList:
    models: List[ModelMetadata]


def register_model_metadata_from_path(path: str) -> None:
    """Register model configurations from the given path."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    # Using dacite instead of cattrs because cattrs doesn't have a default
    # serialization format for dates
    model_metadata_list = dacite.from_dict(ModelMetadataList, raw)
    for model_metadata in model_metadata_list.models:
        model = Model(
            group="none",  # TODO: Group should be part of model deployment, not model
            name=model_metadata.name,
            tags=model_metadata.tags,
        )
        MODEL_NAME_TO_MODEL[model_metadata.name] = model
        ALL_MODELS.append(model)
