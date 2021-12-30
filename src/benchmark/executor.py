from dacite import from_dict
import dataclasses
import requests
from urllib.parse import urlencode
from dataclasses import dataclass, replace
import json

from common.hierarchical_logger import htrack, hlog
from common.request import Request, RequestResult
from common.authentication import Authentication
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
        def process(request_state: RequestState) -> RequestState:
            result = make_request(self.execution_spec.auth, self.execution_spec.url, request_state.request)
            return replace(request_state, result=result)

        request_states = list(map(process, scenario_state.request_states))
        return ScenarioState(scenario_state.adapter_spec, request_states)
