from typing import Optional
from dataclasses import dataclass, replace

from helm.common.general import parallel_map
from helm.common.hierarchical_logger import htrack, hlog
from helm.common.request import RequestResult, Sequence
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
    # If non-empty, URL of the proxy server we send requests to (e.g., http://localhost:1959).
    url: Optional[str]

    # Pass into the service
    auth: Authentication

    # Path where API credentials and cache is stored.
    # This path is the same as `--base-path` when launching the proxy server (see server.py).
    # Required when url is not set.
    local_path: Optional[str]

    # How many threads to have at once
    parallelism: int

    # Whether to skip execution
    dry_run: bool = False

    # Where to store data for the cache.
    # If None, the cache will be disabled.
    # If a path to the directory, this specifies the directory in which the SQLite cache will store files.
    # If a MongoDB URI starting with , this specifies the MongoDB database to be used by the MongoDB cache.
    cache_path: Optional[str] = None


class Executor:
    """
    An `Executor` takes a `ScenarioState` which has a bunch of requests.
    Issue them to the API and return the results.
    """

    def __init__(self, execution_spec: ExecutionSpec):
        self.execution_spec = execution_spec

        self.service: Service
        if execution_spec.url:
            hlog(f"Running using remote API proxy server: {execution_spec.url}")
            self.service = RemoteService(execution_spec.url)
        elif execution_spec.local_path:
            hlog(f"Running in local mode with base path: {execution_spec.local_path}")
            self.service = ServerService(
                base_path=execution_spec.local_path,
                root_mode=True,
                cache_path=execution_spec.cache_path,
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
        return ScenarioState(scenario_state.adapter_spec, request_states)

    def process(self, state: RequestState) -> RequestState:
        try:
            result: RequestResult = self.service.make_request(self.execution_spec.auth, state.request)
        except Exception as e:
            raise ExecutorError(f"{str(e)} Request: {state.request}") from e
        if not result.success:
            if result.error_flags and not result.error_flags.is_fatal:
                hlog(f"WARNING: Non-fatal error treated as empty completion: {result.error}")
                result.completions = [Sequence(text="", logprob=0, tokens=[])]
            else:
                raise ExecutorError(f"{str(result.error)} Request: {state.request}")
        return replace(state, result=result)
