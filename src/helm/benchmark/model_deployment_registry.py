from typing import Dict, Optional, List
from dataclasses import dataclass

import cattrs
import yaml

from helm.common.hierarchical_logger import hlog
from helm.common.object_spec import ObjectSpec
from helm.benchmark.model_metadata_registry import (
    ModelMetadata,
    get_model_metadata,
    get_unknown_model_metadata,
    register_model_metadata,
)


class ClientSpec(ObjectSpec):
    pass


class WindowServiceSpec(ObjectSpec):
    pass


@dataclass(frozen=True)
class ModelDeployment:
    """
    A model deployment is an accessible instance of this model (e.g., a hosted endpoint).
    A model can have multiple model deployments.
    """

    name: str
    """Name of the model deployment. Usually formatted as "<hosting_group>/<engine_name>".
    Example: "huggingface/t5-11b"."""

    client_spec: ClientSpec
    """Specification for instantiating the client for this model deployment."""

    model_name: Optional[str] = None
    """Name of the model that this model deployment is for. Refers to the field "name" in the Model class.
    If unset, defaults to the same value as `name`."""

    tokenizer_name: Optional[str] = None
    """Tokenizer for this model deployment. If unset, auto-inferred by the WindowService."""

    window_service_spec: Optional[WindowServiceSpec] = None
    """Specification for instantiating the window service for this model deployment."""

    max_sequence_length: Optional[int] = None
    """Maximum sequence length for this model deployment."""

    max_request_length: Optional[int] = None
    """Maximum request length for this model deployment.
    If unset, defaults to the same value as max_sequence_length."""

    max_sequence_and_generated_tokens_length: Optional[int] = None
    """The max length of the model input and output tokens.
    Some models (like Anthropic/Claude and Megatron) have a specific limit sequence length + max_token.
    If unset, defaults to INT_MAX (i.e., no limit)."""

    deprecated: bool = False
    """Whether this model deployment is deprecated."""

    @property
    def host_organization(self) -> str:
        """
        Extracts the host group from the model deployment name.
        Example: "huggingface" from "huggingface/t5-11b"
        This can be different from the creator organization (for example "together")
        """
        return self.name.split("/")[0]

    @property
    def engine(self) -> str:
        """
        Extracts the model engine from the model deployment name.
        Example: 'ai21/j1-jumbo' => 'j1-jumbo'
        """
        return self.name.split("/")[1]

    def __post_init__(self):
        if not self.model_name:
            object.__setattr__(self, "model_name", self.name)


@dataclass(frozen=True)
class ModelDeployments:
    model_deployments: List[ModelDeployment]


ALL_MODEL_DEPLOYMENTS: List[ModelDeployment] = []
DEPLOYMENT_NAME_TO_MODEL_DEPLOYMENT: Dict[str, ModelDeployment] = {
    deployment.name: deployment for deployment in ALL_MODEL_DEPLOYMENTS
}


def register_model_deployment(model_deployment: ModelDeployment) -> None:
    DEPLOYMENT_NAME_TO_MODEL_DEPLOYMENT[model_deployment.name] = model_deployment
    ALL_MODEL_DEPLOYMENTS.append(model_deployment)

    model_name: str = model_deployment.model_name or model_deployment.name

    model_metadata: ModelMetadata
    try:
        model_metadata = get_model_metadata(model_name)
    except ValueError:
        hlog(
            f"WARNING: Could not find model metadata for model {model_name} of model deployment {model_deployment.name}"
        )
        model_metadata = get_unknown_model_metadata(model_name)
        register_model_metadata(model_metadata)
    deployment_names: List[str] = model_metadata.deployment_names or [model_metadata.name]
    if model_deployment.name not in deployment_names:
        if model_metadata.deployment_names is None:
            model_metadata.deployment_names = []
        model_metadata.deployment_names.append(model_deployment.name)


def register_model_deployments_from_path(path: str) -> None:
    hlog(f"Reading model deployments from {path}...")
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    model_deployments: ModelDeployments = cattrs.structure(raw, ModelDeployments)
    for model_deployment in model_deployments.model_deployments:
        register_model_deployment(model_deployment)


def get_model_deployment(name: str, warn_deprecated: bool = False) -> ModelDeployment:
    if name not in DEPLOYMENT_NAME_TO_MODEL_DEPLOYMENT:
        raise ValueError(f"Model deployment {name} not found")
    deployment: ModelDeployment = DEPLOYMENT_NAME_TO_MODEL_DEPLOYMENT[name]
    if deployment.deprecated and warn_deprecated:
        hlog(f"WARNING: DEPLOYMENT Model deployment {name} is deprecated")
    return deployment


def get_model_deployment_host_organization(name: str) -> str:
    """Return the host organization name based on the model deployment name.

    Example: "huggingface/t5-11b" -> "huggingface"""
    deployment: ModelDeployment = get_model_deployment(name)
    return deployment.host_organization


def get_model_names_with_tokenizer(tokenizer_name: str) -> List[str]:
    """Return the names of all models with the given tokenizer."""
    deployments: List[ModelDeployment] = [
        deployment for deployment in ALL_MODEL_DEPLOYMENTS if deployment.tokenizer_name == tokenizer_name
    ]
    return [deployment.model_name or deployment.name for deployment in deployments]
