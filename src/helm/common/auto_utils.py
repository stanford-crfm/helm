"""Functions used in both AutoClient and AutoTokenizer."""

import os
from typing import Any, Mapping, Optional

from helm.common.cache import CacheConfig, MongoCacheConfig, SqliteCacheConfig
from helm.common.hierarchical_logger import hlog


def build_cache_config(cache_path: str, mongo_uri: str, organization: str) -> CacheConfig:
    if mongo_uri:
        return MongoCacheConfig(mongo_uri, collection_name=organization)

    client_cache_path: str = os.path.join(cache_path, f"{organization}.sqlite")
    # TODO: Allow setting CacheConfig.follower_cache_path from a command line flag.
    return SqliteCacheConfig(client_cache_path)


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
