import os
from typing import Optional, List
from dataclasses import dataclass, field
from datetime import date

import dacite
import yaml

from helm.proxy.models import ALL_MODELS, MODEL_NAME_TO_MODEL, Model


MODEL_METADATA_FILE = "model_metadata.yaml"


@dataclass(frozen=True)
class ModelMetadata:
    name: str
    """Name of the model e.g. "meta/llama-2"."""

    creator_organization: Optional[str] = None
    """Organization that originally created the model (e.g. "meta")."""

    access: Optional[str] = None
    """How this model is available (e.g., limited).

    If there are multiple deployments, this should be the most permissive access across
    all deployments."""

    todo: bool = False
    """Whether we have yet to evaluate this model."""

    release_date: Optional[date] = None
    """When the model was released."""

    num_parameters: Optional[int] = None
    """The number of model parameters.

    This should be a string as the number of parameters is usually a round number (175B),
    but we set it as an int for plotting purposes."""

    tags: List[str] = field(default_factory=list)
    """"""


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


def maybe_register_model_metadata_from_base_path(base_path: str) -> None:
    """Register model metadata from prod_env/model_metadata.yaml"""
    path = os.path.join(base_path, MODEL_METADATA_FILE)
    if os.path.exists(path):
        register_model_metadata_from_path(path)
