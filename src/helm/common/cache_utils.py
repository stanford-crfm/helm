"""Functions used for caching."""

import os

from helm.common.cache import CacheConfig, MongoCacheConfig, SqliteCacheConfig


def build_cache_config(cache_path: str, mongo_uri: str, organization: str) -> CacheConfig:
    if mongo_uri:
        return MongoCacheConfig(mongo_uri, collection_name=organization)

    client_cache_path: str = os.path.join(cache_path, f"{organization}.sqlite")
    # TODO: Allow setting CacheConfig.follower_cache_path from a command line flag.
    return SqliteCacheConfig(client_cache_path)
