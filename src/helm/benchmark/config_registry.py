from helm.benchmark.model_deployment_registry import register_deployments_if_not_already_registered
from helm.benchmark.model_metadata_registry import register_metadatas_if_not_already_registered
from helm.benchmark.tokenizer_config_registry import register_tokenizers_if_not_already_registered

HELM_REGISTERED: bool = False


def register_helm_configurations(base_path: str = "prod_env"):
    global HELM_REGISTERED
    if not HELM_REGISTERED:
        register_metadatas_if_not_already_registered(base_path)
        register_tokenizers_if_not_already_registered(base_path)
        register_deployments_if_not_already_registered(base_path)
        HELM_REGISTERED = True
