from typing import Optional
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from tqdm import tqdm

from common.general import format_tags
from common.hierarchical_logger import htrack, hlog
from common.request import RequestResult
from common.authentication import Authentication
from proxy.remote_service import RemoteService
from .adapter import RequestState, ScenarioState


@dataclass(frozen=True)
class ExecutionSpec:
    # Where the service lives (e.g., http://localhost:1959)
    url: str

    # Pass into the service
    auth: Authentication

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
        self.remote_service = RemoteService(self.execution_spec.url)

    @htrack(None)
    def execute(self, scenario_state: ScenarioState) -> ScenarioState:
        if self.execution_spec.dry_run:
            hlog("Skipped execution.")
            return scenario_state

        def render_instance(state: RequestState) -> str:
            instance = state.instance
            gold_output: Optional[str] = None
            if instance.first_correct_reference is not None:
                gold_output = instance.first_correct_reference.output

            pred_output: Optional[str] = None
            if state.result is not None:
                pred_output = state.result.completions[0].text

            if state.output_mapping is not None and pred_output is not None:
                pred_output = state.output_mapping.get(pred_output.strip())

            tags_str = format_tags(instance.tags)
            correct_str = "CORRECT" if gold_output == pred_output else "WRONG"
            return (
                f'[{tags_str}] "{instance.input[:100]}" => "{gold_output}", predicted "{pred_output}" [{correct_str}]'
            )

        def process(state: RequestState) -> RequestState:
            result: RequestResult = self.remote_service.make_request(self.execution_spec.auth, state.request)
            state = replace(state, result=result)
            tqdm.write(render_instance(state))
            return state

        with ThreadPoolExecutor(max_workers=self.execution_spec.parallelism) as executor:
            # Run `process` on each request state
            request_states = list(
                tqdm(executor.map(process, scenario_state.request_states), total=len(scenario_state.request_states))
            )

        # Output instances and their predictions
        for i, request_state in enumerate(request_states):
            hlog(f"{i+1}/{len(scenario_state.request_states)}: {render_instance(request_state)}")

        hlog(f"Processed {len(request_states)} requests")
        return ScenarioState(scenario_state.adapter_spec, request_states)
