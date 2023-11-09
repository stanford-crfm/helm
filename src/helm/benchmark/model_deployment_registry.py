import os
from typing import Dict, Optional, List
from dataclasses import dataclass

import cattrs
import yaml

from helm.common.hierarchical_logger import hlog
from helm.common.object_spec import ObjectSpec
from helm.benchmark.model_metadata_registry import (
    ModelMetadata,
    get_model_metadata,
    register_metadatas_if_not_already_registered,
    CONFIG_PATH,
)
from helm.benchmark.tokenizer_config_registry import register_tokenizers_if_not_already_registered


MODEL_DEPLOYMENTS_FILE: str = "model_deployments.yaml"
DEPLOYMENTS_REGISTERED: bool = False
HELM_REGISTERED: bool = False


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
    # hlog(f"Registered model deployment {model_deployment.name}")
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
        raise ValueError(f"Model deployment {model_deployment.name} has no corresponding model metadata")


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
    register_deployments_if_not_already_registered()
    if name not in DEPLOYMENT_NAME_TO_MODEL_DEPLOYMENT:
        raise ValueError(f"Model deployment {name} not found")
    deployment: ModelDeployment = DEPLOYMENT_NAME_TO_MODEL_DEPLOYMENT[name]
    if deployment.deprecated:
        hlog(f"WARNING: Model deployment {name} is deprecated")
    return deployment


# TODO: Figure out when we can remove this.
#
# This is used in 2 contexts:
# 1. Backward compatibility. We can remove this when we no longer want to offer backwards compatibility
#    for model names that are now moved to model deployments (PR #1903).
# 2. Support use cases like "model=text". It was decided that "model_deployment=text" would not be
#    supportedfor now as it coudl evaluate several deployments for the same model. This use case will
#    still need to be supported after we remove backward compatibility.


def get_deployment_name_from_model_arg(
    model_arg: str, can_return_empy: bool = False, warn_arg_deprecated: bool = True
) -> str:
    """Returns a valid model deployment name corresponding to the given model arg.
    This is used as a backwards compatibility layer for model names that are now moved to model deployments.
    Example: "anthropic/claude-v1.3" => "anthropic/claude-v1.3"
    Example: "meta/llama-7b" => "together/llama-7b"

    The process to find a model deployment name is as follows:
    1. If there is a model deployment with the same name as the model arg, use it.
    2. If there is at least one deployment for the model, use the first one that is available.
    3. If there are no deployments for the model, raise an error.

    This function will also try to find a model deployment name that is not deprecated.
    If there are no non-deprecated deployments, it will return the first deployment (even if it's deprecated),
    unless can_return_empy is True, in which case it will return an empty string.

    If warn_arg_deprecated is True, this function will print a warning if the model deployment name is not the same
    as the model arg. This is to remind the user that the model name is deprecated and should be replaced with
    the model deployment name (in their config).

    Args:
        model_arg: The model arg to convert to a model deployment name.
        can_return_empy: Whether to return an empty string if some model deployments were found but all are deprecated.
        warn_arg_deprecated: Whether to print a warning if the model deployment name is not the same as the model arg.
    """

    # Register model deployments if not already registered.
    register_deployments_if_not_already_registered()

    # If there is a model deployment with the same name as the model arg, use it.
    if model_arg in DEPLOYMENT_NAME_TO_MODEL_DEPLOYMENT:
        deployment: ModelDeployment = DEPLOYMENT_NAME_TO_MODEL_DEPLOYMENT[model_arg]
        if deployment.deprecated and can_return_empy:
            hlog(f"WARNING: Model deployment {model_arg} is deprecated")
            return ""
        return deployment.name

    # If there is at least one deployment for the model, use the first one that is available.
    available_deployments: List[ModelDeployment] = [
        deployment for deployment in ALL_MODEL_DEPLOYMENTS if deployment.model_name == model_arg
    ]
    if len(available_deployments) > 0:
        available_deployment_names: List[str] = [deployment.name for deployment in available_deployments]
        if warn_arg_deprecated:
            hlog("WARNING: Model name is deprecated. Please use the model deployment name instead.")
            hlog(f"Available model deployments for model {model_arg}: {available_deployment_names}")

        # Additionally, if there is a non-deprecated deployment, use it.
        non_deprecated_deployments: List[ModelDeployment] = [
            deployment for deployment in available_deployments if not deployment.deprecated
        ]
        if len(non_deprecated_deployments) > 0:
            chosen_deployment = non_deprecated_deployments[0]
        # There are no non-deprecated deployments, so there are two options:
        # 1. If we can return an empty string, return it. (no model deployment is available)
        # 2. If we can't return an empty string, return the first deployment (even if it's deprecated).
        elif can_return_empy:
            return ""
        else:
            hlog(f"WARNING: All model deployments for model {model_arg} are deprecated.")
            chosen_deployment = available_deployments[0]
        if warn_arg_deprecated:
            hlog(
                f"Choosing {chosen_deployment.name} (the first one) as "
                f"the default model deployment for model {model_arg}"
            )
            hlog("If you want to use a different model deployment, please specify it explicitly.")
        return chosen_deployment.name

    raise ValueError(f"Model deployment {model_arg} not found")


def get_model_deployments_by_host_group(host_group: str) -> List[str]:
    """
    Gets models by host group.
    Example:   together   =>   TODO(PR)
    """
    register_deployments_if_not_already_registered()
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
    register_deployments_if_not_already_registered()
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
    register_deployments_if_not_already_registered()
    deployments: List[ModelDeployment] = [
        deployment for deployment in ALL_MODEL_DEPLOYMENTS if deployment.tokenizer_name == tokenizer_name
    ]
    return [deployment.model_name or deployment.name for deployment in deployments]


def register_deployments_if_not_already_registered() -> None:
    global DEPLOYMENTS_REGISTERED
    if not DEPLOYMENTS_REGISTERED:
        maybe_register_model_deployments_from_base_path(CONFIG_PATH)
        DEPLOYMENTS_REGISTERED = True


def maybe_register_helm():
    global HELM_REGISTERED
    if not HELM_REGISTERED:
        register_metadatas_if_not_already_registered()
        register_tokenizers_if_not_already_registered()
        register_deployments_if_not_already_registered()
        HELM_REGISTERED = True
