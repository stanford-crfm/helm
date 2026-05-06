# Model request API

HELM represents model calls with the shared `Request` and `RequestResult`
dataclasses in `helm.common.request`. These classes are the common boundary
between scenarios, clients, the proxy server, and cached raw results.

Use this API when you need to make a model request from Python code or inspect
the exact request and response fields used by HELM runs.

## Request and response formats

::: helm.common.request.Request

::: helm.common.request.RequestResult

::: helm.common.request.GeneratedOutput

::: helm.common.request.Token

## Making a request through `RemoteService`

Use `RemoteService` when you want to call a running HELM proxy server. The
request is serialized and sent to the proxy, and the response is deserialized
as a `RequestResult`.

```python
from helm.common.authentication import Authentication
from helm.common.request import Request
from helm.proxy.services.remote_service import RemoteService

service = RemoteService("http://localhost:1959")
auth = Authentication(api_key="YOUR_HELM_API_KEY")

request = Request(
    model_deployment="openai/gpt-4o-mini",
    model="openai/gpt-4o-mini",
    prompt="Explain HELM in one sentence.",
    max_tokens=64,
    temperature=0.0,
)

result = service.make_request(auth, request)

if result.success:
    print(result.completions[0].text)
else:
    print(result.error)
```

See `helm.proxy.services.remote_service.RemoteService` for the complete
remote-service interface.

## Making a request through `AutoClient`

Use `AutoClient` when you want HELM to select the concrete client from the
`model_deployment` field and use local credentials directly. `AutoClient`
requires a credentials mapping, a file storage path, and a cache backend
configuration.

```python
from helm.clients.auto_client import AutoClient
from helm.common.cache import CacheConfig
from helm.common.cache_backend_config import BlackHoleCacheBackendConfig
from helm.common.request import Request

client = AutoClient(
    credentials={"openaiApiKey": "YOUR_OPENAI_API_KEY"},
    file_storage_path="prod_env/cache",
    cache_backend_config=BlackHoleCacheBackendConfig(
        cache_config=CacheConfig(cache_path="prod_env/cache")
    ),
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

## Choosing between the two

- Use `RemoteService` when a HELM proxy server owns credentials, caching, and
  request routing.
- Use `AutoClient` when your Python process owns the provider credentials and
  should dispatch directly to provider-specific clients.
- In both cases, call `request.validate()` before dispatch if you construct
  requests dynamically and want to fail early on incompatible prompt fields.
