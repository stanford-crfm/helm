from helm.benchmark.model_deployment_registry import register_deployments_if_not_already_registered
from helm.benchmark.model_metadata_registry import register_metadatas_if_not_already_registered
from helm.benchmark.tokenizer_config_registry import register_tokenizers_if_not_already_registered

HELM_REGISTERED: bool = False


def register_helm_configurations():
    global HELM_REGISTERED
    if not HELM_REGISTERED:
        register_metadatas_if_not_already_registered()
        register_tokenizers_if_not_already_registered()
        register_deployments_if_not_already_registered()
        HELM_REGISTERED = True
