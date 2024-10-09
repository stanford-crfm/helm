import os
import importlib_resources as resources

from helm.benchmark.model_deployment_registry import register_model_deployments_from_path
from helm.benchmark.model_metadata_registry import register_model_metadata_from_path
from helm.benchmark.tokenizer_config_registry import register_tokenizer_configs_from_path
from helm.benchmark.runner_config_registry import register_runner_config_from_path


MODEL_METADATA_FILE: str = "model_metadata.yaml"
TOKENIZER_CONFIGS_FILE: str = "tokenizer_configs.yaml"
MODEL_DEPLOYMENTS_FILE: str = "model_deployments.yaml"
RUNNER_CONFIG_FILE: str = "runner_config.yaml"

CONFIG_PACKAGE = "helm.config"


def register_configs_from_directory(dir_path: str) -> None:
    model_metadata_path = os.path.join(dir_path, MODEL_METADATA_FILE)
    if os.path.isfile(model_metadata_path):
        register_model_metadata_from_path(model_metadata_path)

    tokenizer_configs_path = os.path.join(dir_path, TOKENIZER_CONFIGS_FILE)
    if os.path.isfile(tokenizer_configs_path):
        register_tokenizer_configs_from_path(tokenizer_configs_path)

    model_deployments_path = os.path.join(dir_path, MODEL_DEPLOYMENTS_FILE)
    if os.path.isfile(model_deployments_path):
        register_model_deployments_from_path(model_deployments_path)

    runner_config_path = os.path.join(dir_path, RUNNER_CONFIG_FILE)
    if os.path.isfile(runner_config_path):
        register_runner_config_from_path(runner_config_path)


def register_builtin_configs_from_helm_package() -> None:
    package_path = str(resources.files(CONFIG_PACKAGE))
    register_configs_from_directory(package_path)
