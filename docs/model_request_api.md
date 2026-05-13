# Model request API

HELM represents model calls with the shared `Request` and `RequestResult`
dataclasses in `helm.common.request`. These classes are the common boundary
between scenarios, clients, local execution, and cached raw results.

Use this API when you need to make a model request from Python code or inspect
the exact request and response fields used by HELM runs.

## Request and response formats

::: helm.common.request.Request

::: helm.common.request.RequestResult

::: helm.common.request.GeneratedOutput

::: helm.common.request.Token

## Making a local request through `AutoClient`

Use `AutoClient` when you want HELM to select the concrete client from the
`model_deployment` field and use local credentials directly. This is the
recommended path for making model requests from Python code. `AutoClient`
requires a credentials mapping, a file storage path, and a cache backend
configuration. The example below uses `BlackHoleCacheBackendConfig`, which
does not persist cache entries.

```python
from helm.clients.auto_client import AutoClient
from helm.common.cache_backend_config import BlackHoleCacheBackendConfig
from helm.common.request import Request

client = AutoClient(
    credentials={"openaiApiKey": "YOUR_OPENAI_API_KEY"},
    file_storage_path="prod_env/cache",
    cache_backend_config=BlackHoleCacheBackendConfig(),
)

request = Request(
    model_deployment="openai/gpt-4o-mini",
    model="openai/gpt-4o-mini",
    prompt="Explain HELM in one sentence.",
    max_tokens=64,
    temperature=0.0,
)

result = client.make_request(request)

if result.success:
    print(result.completions[0].text)
else:
    print(result.error)
```

See `helm.clients.auto_client.AutoClient` for the complete local-client
interface.

## Using a persistent cache

Use `SqliteCacheBackendConfig` when you want HELM to persist request results
locally:

```python
from helm.common.cache_backend_config import SqliteCacheBackendConfig

cache_backend_config = SqliteCacheBackendConfig(path="prod_env/cache")
```

Pass this value as `cache_backend_config` when constructing `AutoClient`.

Call `request.validate()` before dispatch if you construct requests
dynamically and want to fail early on incompatible prompt fields.
