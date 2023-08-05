"""This allows users to register additional models via configuration file.

TODO(#1673): Support adding other kinds of models besides AI21.
"""

from typing import Any, Dict, Iterable, Optional
from dataclasses import dataclass

import cattrs

from helm.common.general import parse_hocon
from helm.proxy.models import ALL_MODELS, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, MODEL_NAME_TO_MODEL, TEXT_MODEL_TAG, Model


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for a registered model."""

    model_type: str
    """Name of the client type."""

    # TODO(#1673): Add tokenizer name and sequence length fields.

    args: Optional[Dict[str, Any]] = None
    """Configuration for the model"""


_name_to_model_config: Dict[str, ModelConfig] = {}


def register_model_configs_from_path(path: str) -> ModelConfig:
    """Register model configurations from the given path."""
    global _name_to_model_config
    with open(path, "r") as f:
        raw = parse_hocon(f.read())
        name_to_model_config = cattrs.structure(raw, Dict[str, ModelConfig])
        _name_to_model_config.update(name_to_model_config)
        for name, model_config in name_to_model_config.items():
            model = Model(
                group=model_config.model_type,
                name=name,
                tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG],
            )
            MODEL_NAME_TO_MODEL[name] = model
            ALL_MODELS.append(model)


def get_model_config(name: str) -> Optional[ModelConfig]:
    """Return the ModelConfig for the given model."""
    global _name_to_model_config
    return _name_to_model_config.get(name)
