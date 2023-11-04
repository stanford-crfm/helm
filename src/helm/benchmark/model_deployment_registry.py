import os
from typing import Dict, Optional, List
from dataclasses import dataclass
from datetime import date

import cattrs
import yaml

from helm.common.hierarchical_logger import hlog
from helm.common.object_spec import ObjectSpec
from helm.benchmark.model_metadata_registry import (
    ModelMetadata,
    register_model_metadata,
    get_model_metadata,
    TEXT_MODEL_TAG,
    FULL_FUNCTIONALITY_TEXT_MODEL_TAG,
)


MODEL_DEPLOYMENTS_FILE = "model_deployments.yaml"


class ClientSpec(ObjectSpec):
    pass


class WindowServiceSpec(ObjectSpec):
    pass


@dataclass(frozen=True)
class ModelDeployment:
    """A model deployment is an accessible instance of this model (e.g. a hosted endpoint).

    A model can have multiple model deployments."""

    # Name of the model deployment.
    # Usually formatted as "<hosting_group>/<engine_name>"
    # Example: "huggingface/t5-11b"
    name: str

    # Specification for instantiating the client for this model deployment.
    client_spec: ClientSpec

    # Name of the model that this model deployment is for.
    # Refers to the field "name" in the Model class.
    # If unset, defaults to the same value as `name`.
    model_name: Optional[str] = None

    # Tokenizer for this model deployment.
    # If unset, auto-inferred by the WindowService.
    tokenizer_name: Optional[str] = None

    # Specification for instantiating the window service for this model deployment.
    window_service_spec: Optional[WindowServiceSpec] = None

    # Maximum sequence length for this model deployment.
    max_sequence_length: Optional[int] = None

    # Maximum request length for this model deployment.
    # If unset, defaults to the same value as max_sequence_length.
    max_request_length: Optional[int] = None

    # The max length of the model input and output tokens.
    # Some models (like Anthropic/Claude and Megatron) have a specific limit sequence length + max_token.
    # If unset, defaults to INT_MAX (i.e. bo limit).
    max_sequence_and_generated_tokens_length: Optional[int] = None

    # Whether this model deployment is deprecated.
    deprecated: bool = False

    @property
    def host_group(self) -> str:
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


@dataclass(frozen=True)
class ModelDeployments:
    model_deployments: List[ModelDeployment]


ALL_MODEL_DEPLOYMENTS: List[ModelDeployment] = []
DEPLOYMENT_NAME_TO_MODEL_DEPLOYMENT: Dict[str, ModelDeployment] = {
    deployment.name: deployment for deployment in ALL_MODEL_DEPLOYMENTS
}


# ===================== REGISTRATION FUNCTIONS ==================== #
def register_model_deployment(model_deployment: ModelDeployment) -> None:
    hlog(f"Registered model deployment {model_deployment.name}")
    DEPLOYMENT_NAME_TO_MODEL_DEPLOYMENT[model_deployment.name] = model_deployment
    ALL_MODEL_DEPLOYMENTS.append(model_deployment)

    model_name: str = model_deployment.model_name or model_deployment.name

    try:
        model_metadata: ModelMetadata = get_model_metadata(model_name)
        deployment_names: List[str] = model_metadata.deployment_names or [model_metadata.name]
        if model_deployment.name not in deployment_names:
            if model_metadata.deployment_names is None:
                model_metadata.deployment_names = []
            model_metadata.deployment_names.append(model_deployment.name)
    except ValueError:
        # No model metadata exists for this model name.
        # Register a default model metadata.
        model_metadata = ModelMetadata(
            name=model_name,
            display_name=model_name,
            description="",
            access="limited",
            num_parameters=-1,
            release_date=date.today(),
            creator_organization_name="unknown",
            tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG],
            deployment_names=[model_deployment.name],
        )
        register_model_metadata(model_metadata)
        hlog(f"Registered default metadata for model {model_name}")


def register_model_deployments_from_path(path: str) -> None:
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


# ===================== UTIL FUNCTIONS ==================== #
def get_model_deployment(name: str) -> ModelDeployment:
    if name not in DEPLOYMENT_NAME_TO_MODEL_DEPLOYMENT:
        raise ValueError(f"Model deployment {name} not found")
    deployment: ModelDeployment = DEPLOYMENT_NAME_TO_MODEL_DEPLOYMENT[name]
    if deployment.deprecated:
        hlog(f"WARNING: Model deployment {name} is deprecated")
    return deployment


def get_model_deployments_by_host_group(host_group: str) -> List[str]:
    """
    Gets models by host group.
    Example:   together   =>   TODO(PR)
    """
    return [deployment.name for deployment in ALL_MODEL_DEPLOYMENTS if deployment.host_group == host_group]


def get_model_deployment_host_group(name: str) -> str:
    """
    Extracts the host group from the model deployment name.
    Example: "huggingface/t5-11b" => "huggingface"
    """
    deployment: ModelDeployment = get_model_deployment(name)
    return deployment.host_group


def get_default_deployment_for_model(model_metadata: ModelMetadata) -> ModelDeployment:
    """
    Given a model_metadata, returns the default model deployment.
    The default model deployment for a model is either the deployment
    with the same name as the model, or the first deployment for that model.

    TODO: Make this logic more complex.
    For example if several deplyments are available but only some can be used
    given the API keys present, then we should choose the one that can be used.
    """
    if model_metadata.name in DEPLOYMENT_NAME_TO_MODEL_DEPLOYMENT:
        return DEPLOYMENT_NAME_TO_MODEL_DEPLOYMENT[model_metadata.name]
    elif model_metadata.deployment_names is not None and len(model_metadata.deployment_names) > 0:
        deployment_name: str = model_metadata.deployment_names[0]
        if deployment_name in DEPLOYMENT_NAME_TO_MODEL_DEPLOYMENT:
            return DEPLOYMENT_NAME_TO_MODEL_DEPLOYMENT[deployment_name]
        raise ValueError(f"Model deployment {deployment_name} not found")
    raise ValueError(f"No default model deployment for model {model_metadata.name}")


def get_metadata_for_deployment(deployment_name: str) -> ModelMetadata:
    """
    Given a deployment name, returns the corresponding model metadata.
    """
    deployment: ModelDeployment = get_model_deployment(deployment_name)
    return get_model_metadata(deployment.model_name or deployment.name)


def get_model_names_with_tokenizer(tokenizer_name: str) -> List[str]:
    """Get all the name of the models with tokenizer `tokenizer_name`."""
    deployments: List[ModelDeployment] = [
        deployment for deployment in ALL_MODEL_DEPLOYMENTS if deployment.tokenizer_name == tokenizer_name
    ]
    return [deployment.model_name or deployment.name for deployment in deployments]
