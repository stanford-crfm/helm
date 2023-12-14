"""Functions used for caching."""

import os
from typing import Optional

from helm.common.cache import BlackHoleCacheConfig, CacheConfig, MongoCacheConfig, SqliteCacheConfig


def build_cache_config(cache_path: Optional[str], organization: str) -> CacheConfig:
    if cache_path is None:
        return BlackHoleCacheConfig()
    elif cache_path.startswith("mongodb:"):
        return MongoCacheConfig(cache_path, collection_name=organization)
    else:
        return SqliteCacheConfig(os.path.join(cache_path, f"{organization}.sqlite"))
