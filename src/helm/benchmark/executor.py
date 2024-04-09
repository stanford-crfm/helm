from typing import Optional
from dataclasses import dataclass, replace
from helm.common.cache_backend_config import (
    CacheBackendConfig,
    BlackHoleCacheBackendConfig,
    MongoCacheBackendConfig,
    SqliteCacheBackendConfig,
)

from helm.common.general import parallel_map
from helm.common.hierarchical_logger import htrack, hlog
from helm.common.request import RequestResult, GeneratedOutput
from helm.common.authentication import Authentication
from helm.proxy.services.remote_service import RemoteService
from helm.proxy.services.server_service import ServerService
from helm.proxy.services.service import Service
from helm.benchmark.adaptation.scenario_state import ScenarioState
from helm.benchmark.adaptation.request_state import RequestState


class ExecutorError(Exception):
    pass


@dataclass(frozen=True)
class ExecutionSpec:

    url: Optional[str]
    """If non-empty, URL of the proxy server we send requests to (e.g., http://localhost:1959)."""

    auth: Authentication
    """Authentication that will be passed into the local service, if using the local service."""

    local_path: Optional[str]
    """Path where API credentials and cache is stored.

    This path is the same as `--base-path` when launching the proxy server (see server.py).
    Required when url is not set."""

    parallelism: int
    """How many threads to have at once"""

    dry_run: bool = False
    """Whether to skip execution"""

    sqlite_cache_backend_config: Optional[SqliteCacheBackendConfig] = None
    """If set, SQLite will be used for the cache.

    This specifies the directory in which the SQLite cache will store files.
    At most one of sqlite_cache_backend_config and mongo_cache_backend_config can be set."""

    mongo_cache_backend_config: Optional[MongoCacheBackendConfig] = None
    """If set, MongoDB will be used for the cache.

    This specifies the MongoDB database to be used by the MongoDB cache.
    At most one of sqlite_cache_backend_config and mongo_cache_backend_config can be set."""


class Executor:
    """
    An `Executor` takes a `ScenarioState` which has a bunch of requests.
    Issue them to the API and return the results.
    """

    def __init__(self, execution_spec: ExecutionSpec):
        self.execution_spec = execution_spec

        cache_backend_config: CacheBackendConfig
        if execution_spec.sqlite_cache_backend_config and execution_spec.mongo_cache_backend_config:
            raise ExecutorError("At most one of sqlite_cache_backend_config and mongo_cache_backend_config can be set.")
        elif execution_spec.sqlite_cache_backend_config:
            cache_backend_config = execution_spec.sqlite_cache_backend_config
        elif execution_spec.mongo_cache_backend_config:
            cache_backend_config = execution_spec.mongo_cache_backend_config
        else:
            cache_backend_config = BlackHoleCacheBackendConfig()

        self.service: Service
        if execution_spec.url:
            hlog(f"Running using remote API proxy server: {execution_spec.url}")
            self.service = RemoteService(execution_spec.url)
        elif execution_spec.local_path:
            hlog(f"Running in local mode with base path: {execution_spec.local_path}")
            self.service = ServerService(
                base_path=execution_spec.local_path,
                root_mode=True,
                cache_backend_config=cache_backend_config,
            )
        else:
            raise ValueError("Either the proxy server URL or the local path must be set")

    @htrack(None)
    def execute(self, scenario_state: ScenarioState) -> ScenarioState:
        if self.execution_spec.dry_run:
            hlog("Skipped execution.")
            return scenario_state

        # Do it!
        request_states = parallel_map(
            self.process,
            scenario_state.request_states,
            parallelism=self.execution_spec.parallelism,
        )

        hlog(f"Processed {len(request_states)} requests")
        return ScenarioState(
            adapter_spec=scenario_state.adapter_spec,
            request_states=request_states,
            annotator_specs=scenario_state.annotator_specs,
        )

    def process(self, state: RequestState) -> RequestState:
        try:
            result: RequestResult = self.service.make_request(self.execution_spec.auth, state.request)
        except Exception as e:
            raise ExecutorError(f"{str(e)} Request: {state.request}") from e
        if not result.success:
            if result.error_flags and not result.error_flags.is_fatal:
                hlog(f"WARNING: Non-fatal error treated as empty completion: {result.error}")
                result.completions = [GeneratedOutput(text="", logprob=0, tokens=[])]
            else:
                raise ExecutorError(f"{str(result.error)} Request: {state.request}")
        return replace(state, result=result)
