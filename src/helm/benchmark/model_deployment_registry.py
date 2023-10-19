import os
from typing import Dict, Optional, List
from dataclasses import dataclass

import cattrs
import yaml

from helm.common.hierarchical_logger import hlog
from helm.common.object_spec import ObjectSpec
from helm.proxy.models import ALL_MODELS, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, MODEL_NAME_TO_MODEL, TEXT_MODEL_TAG, Model


MODEL_DEPLOYMENTS_FILE = "model_deployments.yaml"


class ClientSpec(ObjectSpec):
    pass


class WindowServiceSpec(ObjectSpec):
    pass


@dataclass(frozen=True)
class ModelDeployment:
    """A model deployment is an accessible instance of this model (e.g. a hosted endpoint).

    A model can have multiple model deployments."""

    name: str
    """Name of the model deployment."""

    client_spec: ClientSpec
    """Specification for instantiating the client for this model deployment."""

    model_name: Optional[str] = None
    """Name of the model that this model deployment is for.

    If unset, defaults to the the same value as `name`."""

    tokenizer_name: Optional[str] = None
    """Tokenizer for this model deployment.

    If unset, auto-inferred by the WindowService."""

    window_service_spec: Optional[WindowServiceSpec] = None
    """Specification for instantiating the window service for this model deployment"""

    max_sequence_length: Optional[int] = None
    """Maximum sequence length for this model deployment."""

    max_request_length: Optional[int] = None
    """Maximum request length for this model deployment.

    If unset, defaults to the same value as max_sequence_length."""


@dataclass(frozen=True)
class ModelDeployments:
    model_deployments: List[ModelDeployment]


_name_to_model_deployment: Dict[str, ModelDeployment] = {}


def register_model_deployment(model_deployment: ModelDeployment) -> None:
    hlog(f"Registered model deployment {model_deployment.name}")
    _name_to_model_deployment[model_deployment.name] = model_deployment

    # Auto-register a model with this name if none exists
    model_name = model_deployment.model_name or model_deployment.name
    if model_name not in MODEL_NAME_TO_MODEL:
        model = Model(
            group="unknown",
            name=model_name,
            tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG],
        )
        MODEL_NAME_TO_MODEL[model_name] = model
        ALL_MODELS.append(model)
        hlog(f"Registered default metadata for model {model_name}")


def register_model_deployments_from_path(path: str) -> None:
    global _name_to_model_deployment
    hlog(f"Reading model deployments from {path}...")
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    model_deployments: ModelDeployments = cattrs.structure(raw, ModelDeployments)
    for model_deployment in model_deployments.model_deployments:
        register_model_deployment(model_deployment)


def maybe_register_model_deployments_from_base_path(base_path: str) -> None:
    """Register model deployments from prod_env/model_deployments.yaml"""
    path = os.path.join(base_path, MODEL_DEPLOYMENTS_FILE)
    if os.path.exists(path):
        register_model_deployments_from_path(path)


def get_model_deployment(name: str) -> Optional[ModelDeployment]:
    return _name_to_model_deployment.get(name)
