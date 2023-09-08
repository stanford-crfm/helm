import os
from typing import Dict, Optional, List
from dataclasses import dataclass

import cattrs
import yaml

from helm.common.hierarchical_logger import hlog
from helm.common.object_spec import ObjectSpec


MODEL_DEPLOYMENTS_FILE = "model_deployments.yaml"


class ClientSpec(ObjectSpec):
    pass


@dataclass(frozen=True)
class ModelDeployment:
    """A model deployment is an accessible instance of this model (e.g. a hosted endpoint).

    A model can have model deployments."""

    name: str
    """Name of the model deployment."""

    model_name: str
    """Name of the model that this model deployment is for."""

    client_spec: ClientSpec
    """Specification for instantiating the client for this model deployment."""

    max_sequence_length: Optional[int]
    """Maximum sequence length for this model deployment."""

    max_request_length: Optional[int] = None
    """Maximum request length for this model deployment.
    
    If unset, defaults to the same value as max_sequence_length."""

    tokenizer_name: Optional[str] = None
    """Tokenizer for this model deployment.
    
    If unset, defaults to the same value as model_name."""


@dataclass(frozen=True)
class ModelDeployments:
    model_deployments: List[ModelDeployment]


_name_to_model_deployment: Dict[str, ModelDeployment] = {}


def register_model_deployments_from_path(path: str) -> None:
    global _name_to_model_deployment
    hlog(f"Reading model deployments from {path}...")
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    model_deployments: ModelDeployments = cattrs.structure(raw, ModelDeployments)
    for model_deployment in model_deployments.model_deployments:
        _name_to_model_deployment[model_deployment.name] = model_deployment


def maybe_register_model_deployments_from_base_path(base_path: str) -> None:
    path = os.path.join(base_path, MODEL_DEPLOYMENTS_FILE)
    if os.path.exists(path):
        register_model_deployments_from_path(path)


def get_model_deployment(name: str) -> Optional[ModelDeployment]:
    return _name_to_model_deployment.get(name)
