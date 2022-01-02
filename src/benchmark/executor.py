from dacite import from_dict
import dataclasses
from typing import Optional
import requests
from urllib.parse import urlencode
from dataclasses import dataclass, replace
import json

from common.hierarchical_logger import htrack, hlog
from common.request import Request, RequestResult
from common.authentication import Authentication
from .scenario import Instance
from .adapter import RequestState, ScenarioState


@dataclass(frozen=True)
class ExecutionSpec:
    # Where the service lives (e.g., http://localhost:1959)
    url: str

    # Pass into the service
    auth: Authentication

    # How many threads to have at once
    parallelism: int


def make_request(auth: Authentication, url: str, request: Request) -> RequestResult:
    # TODO: replace this by `RemoteService`
    params = {
        "auth": json.dumps(dataclasses.asdict(auth)),
        "request": json.dumps(dataclasses.asdict(request)),
    }
    response = requests.get(url + "/api/request?" + urlencode(params)).json()
    if response.get("error"):
        hlog(response["error"])
    return from_dict(data_class=RequestResult, data=response)


class Executor:
    """
    An `Executor` takes a `ScenarioState` which has a bunch of requests.
    Issue them to the API. and return the results.
    """

    def __init__(self, execution_spec: ExecutionSpec):
        self.execution_spec = execution_spec

    @htrack(None)
    def execute(self, scenario_state: ScenarioState) -> ScenarioState:
        # TODO: make a thread pool to process all of these in parallel up to a certain number
        def render_instance(instance: Instance, pred_output: Optional[str]) -> str:
            tags_str = ",".join(instance.tags)
            gold_output: Optional[str] = None
            if instance.first_correct_reference is not None:
                gold_output = instance.first_correct_reference.output

            if request_state.output_mapping is not None and pred_output is not None:
                pred_output = request_state.output_mapping.get(pred_output.strip())
            correct_str = "CORRECT" if gold_output == pred_output else "WRONG"
            return (
                f'[{tags_str}] "{instance.input[:100]}" => "{gold_output}", predicted "{pred_output}" [{correct_str}]'
            )

        def process(request_state: RequestState) -> RequestState:
            result = make_request(self.execution_spec.auth, self.execution_spec.url, request_state.request)
            prediction = result.completions[0].text
            hlog(f"{i}/{len(scenario_state.request_states)}: {render_instance(instance, prediction)}")
            return replace(request_state, result=result)

        request_states = []
        for i, request_state in enumerate(scenario_state.request_states):
            instance = request_state.instance

            request_states.append(process(request_state))
        hlog(f"Processed {len(request_states)} requests")
        return ScenarioState(scenario_state.adapter_spec, request_states)
