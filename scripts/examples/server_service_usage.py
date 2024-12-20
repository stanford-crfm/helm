import os

from helm.benchmark.config_registry import register_builtin_configs_from_helm_package
from helm.common.authentication import Authentication
from helm.common.cache_backend_config import SqliteCacheBackendConfig
from helm.common.general import ensure_directory_exists
from helm.common.request import Request
from helm.proxy.services.server_service import ServerService
from helm.proxy.services.service import CACHE_DIR


# One-time initialization
register_builtin_configs_from_helm_package()

# Set up SQLite cache
base_path = "prod_env"  # path to prod_env/
sqlite_cache_path = os.path.join(base_path, CACHE_DIR)
ensure_directory_exists(sqlite_cache_path)
cache_backend_config = SqliteCacheBackendConfig(sqlite_cache_path)

# Alternatively, disable the request cache
# from helm.common.cache_backend_config import BlackHoleCacheBackendConfig
# cache_backend_config = BlackHoleCacheBackendConfig()

# API credentials must be placed in prod_env/credentials.conf
service = ServerService(
    base_path=base_path,
    root_mode=True,  # Always set to True
    cache_backend_config=cache_backend_config,
)
authentication = Authentication("")  # Empty for ServerService

request = Request(
    model_deployment="openai/gpt-4o-2024-08-06",
    model="openai/gpt-4o-2024-08-06",
    prompt="Write a haiku about otters.",
)
response = service.make_request(authentication, request)
print(response)

request = Request(
    model="meta/llama-3.1-8b-instruct-turbo",
    model_deployment="together/llama-3.1-8b-instruct-turbo",
    prompt="Write a haiku about otters.",
)
response = service.make_request(authentication, request)
print(response)
