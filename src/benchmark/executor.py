from typing import Optional
from dataclasses import dataclass, replace

from common.general import format_text, parallel_map
from common.hierarchical_logger import htrack, hlog
from common.request import RequestResult
from common.authentication import Authentication
from proxy.services.remote_service import RemoteService
from proxy.services.server_service import ServerService
from proxy.services.service import Service
from .adapter import RequestState, ScenarioState
from .scenarios.scenario import Instance


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
            self.service = ServerService(base_path=execution_spec.local_path, root_mode=True)
        else:
            assert execution_spec.url, "local=False. Need to specify the URL of proxy server (`url`)."
            self.service = RemoteService(self.execution_spec.url)

    @htrack(None)
    def execute(self, scenario_state: ScenarioState) -> ScenarioState:
        if self.execution_spec.dry_run:
            hlog("Skipped execution.")
            return scenario_state

        def render_request_state(state: RequestState) -> str:
            def format_instance(instance: Instance) -> str:
                metadata_str: str = (
                    f"[split={instance.split}, sub_split={instance.sub_split}, "
                    f"id={instance.id}, perturbation={instance.perturbation}]"
                )
                return f"{metadata_str} {format_text(instance.input[:100])}"

            instance: Instance = state.instance
            gold_output: Optional[str] = None
            if instance.first_correct_reference is not None:
                gold_output = instance.first_correct_reference.output

            pred_output: Optional[str] = None
            if state.result is not None:
                pred_output = state.result.completions[0].text

            if state.output_mapping is not None and pred_output is not None:
                pred_output = state.output_mapping.get(pred_output.strip())

            correct_str = "CORRECT" if gold_output == pred_output else "WRONG"
            # Truncate pred_output for a better visualization
            if pred_output is not None:
                pred_output = pred_output[:100]
            return (
                f"{format_instance(instance)} => {format_text(str(gold_output))}, "
                + f"predicted {format_text(str(pred_output))} [{correct_str}]"
            )

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
            raise ExecutorError(f"{str(result.error)} Request: {state.request}")
        return replace(state, result=result)
