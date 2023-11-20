"""Functions used for credentials."""

from typing import Any, Mapping, Optional

from helm.common.hierarchical_logger import hlog


def provide_api_key(
    credentials: Mapping[str, Any], host_organization: str, model: Optional[str] = None
) -> Optional[str]:
    api_key_name = host_organization + "ApiKey"
    if api_key_name in credentials:
        hlog(f"Using host_organization api key defined in credentials.conf: {api_key_name}")
        return credentials[api_key_name]
    if "deployments" not in credentials:
        hlog(
            "WARNING: Could not find key 'deployments' in credentials.conf, "
            f"therefore the API key {api_key_name} should be specified."
        )
        return None
    deployment_api_keys = credentials["deployments"]
    if model is None:
        hlog(f"WARNING: Could not find key '{host_organization}' in credentials.conf and no model provided")
        return None
    if model not in deployment_api_keys:
        hlog(f"WARNING: Could not find key '{model}' under key 'deployments' in credentials.conf")
        return None
    return deployment_api_keys[model]
