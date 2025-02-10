import os

from helm.benchmark.config_registry import register_builtin_configs_from_helm_package
from helm.clients.auto_client import AutoClient
from helm.common.cache_backend_config import SqliteCacheBackendConfig
from helm.common.general import ensure_directory_exists, get_credentials
from helm.common.request import Request
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

credentials = get_credentials(base_path)

# Alternatively, specify credentials directly:
# credentials = {
#     "openaiApiKey": "your_openai_key_here",
#     "togetherApiKey": "your_together_key_here",
# }

file_storage_path = os.path.join(base_path, CACHE_DIR)
client = AutoClient(credentials, file_storage_path, cache_backend_config)

request = Request(
    model_deployment="openai/gpt-4o-2024-08-06",
    model="openai/gpt-4o-2024-08-06",
    prompt="Write a haiku about otters.",
)
response = client.make_request(request)
print(response)

request = Request(
    model="meta/llama-3.1-8b-instruct-turbo",
    model_deployment="together/llama-3.1-8b-instruct-turbo",
    prompt="Write a haiku about otters.",
)
response = client.make_request(request)
print(response)
