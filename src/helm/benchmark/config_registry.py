import os
import importlib_resources as resources

from helm.benchmark.model_deployment_registry import register_model_deployments_from_path
from helm.benchmark.model_metadata_registry import register_model_metadata_from_path
from helm.benchmark.tokenizer_config_registry import register_tokenizer_configs_from_path


MODEL_METADATA_FILE: str = "model_metadata.yaml"
TOKENIZER_CONFIGS_FILE: str = "tokenizer_configs.yaml"
MODEL_DEPLOYMENTS_FILE: str = "model_deployments.yaml"

CONFIG_PACKAGE = "helm.config"


def register_configs_from_directory(dir_path) -> None:
    register_model_metadata_from_path(os.path.join(dir_path, MODEL_METADATA_FILE))
    register_tokenizer_configs_from_path(os.path.join(dir_path, TOKENIZER_CONFIGS_FILE))
    register_model_deployments_from_path(os.path.join(dir_path, MODEL_DEPLOYMENTS_FILE))


def register_configs_from_package() -> None:
    package_path = str(resources.files(CONFIG_PACKAGE))
    register_configs_from_directory(package_path)
