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
    # URL of the proxy server we send requests to (e.g., http://localhost:1959).
    # Required when local=False.
    url: Optional[str]

    # Pass into the service
    auth: Authentication

    # Whether to bypass the proxy server and just run everything locally
    local: bool

    # Path where API credentials and cache is stored.
    # This path is the same as `--base-path` when launching the proxy server (see server.py).
    # Required when local=True.
    local_path: Optional[str]

    # How many threads to have at once
    parallelism: int

    # Whether to skip execution
    dry_run: bool = False

    # URL to the MongoDB database.
    # If non-empty, the MongoDB database will be used for caching instead of SQLite.
    # Example format: mongodb://[username:password@]host1[:port1]/[dbname]
    # For full format, see: https://www.mongodb.com/docs/manual/reference/connection-string/
    mongo_uri: str = ""


class Executor:
    """
    An `Executor` takes a `ScenarioState` which has a bunch of requests.
    Issue them to the API and return the results.
    """

    def __init__(self, execution_spec: ExecutionSpec):
        self.execution_spec = execution_spec

        self.service: Service
        if execution_spec.local:
            assert execution_spec.local_path, "local=True. Need to specify a value for `local_path`."
            hlog(f"Running locally in root mode with local path: {execution_spec.local_path}")
            self.service = ServerService(
                base_path=execution_spec.local_path, root_mode=True, mongo_uri=execution_spec.mongo_uri
            )
        else:
            assert execution_spec.url, "local=False. Need to specify the URL of proxy server (`url`)."
            self.service = RemoteService(self.execution_spec.url)

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
        result: RequestResult = self.service.make_request(self.execution_spec.auth, state.request)
        if not result.success:
            if result.error_flags and not result.error_flags.is_fatal:
                hlog(f"WARNING: Non-fatal error treated as empty completion: {result.error}")
                result.completions = [Sequence(text="", logprob=0, tokens=[])]
            else:
                raise ExecutorError(f"{str(result.error)} Request: {state.request}")
        return replace(state, result=result)
