from collections import defaultdict, OrderedDict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

from helm.benchmark.scenarios.scenario import Instance
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.annotation.annotator import AnnotatorSpec


@dataclass
class ScenarioState:
    """
    A `ScenarioState` represents the output of adaptation.  Contains a set of
    `RequestState` that were created and executed (a `ScenarioState` could be
    pre-execution or post-execution).
    """

    # What strategy we used for adaptation
    adapter_spec: AdapterSpec

    # List of `RequestState`s that were produced by adaptation (and execution)
    request_states: List[RequestState]

    # Annotations to use for this run spec
    annotator_specs: Optional[List[AnnotatorSpec]] = None

    def __post_init__(self):
        # Create derived indices based on `request_states` so it's easier for
        # the `Metric` later to access them.  Two things are produced:
        self.request_state_map: Dict[Tuple[int, Instance, Optional[int]], List[RequestState]] = defaultdict(list)

        # Python doesn't support an ordered set, so use an OrderedDict instead to maintain insertion order
        instances_set: Dict[Instance, None] = OrderedDict()
        for request_state in self.request_states:
            instances_set[request_state.instance] = None
            key = (request_state.train_trial_index, request_state.instance, request_state.reference_index)
            self.request_state_map[key].append(request_state)
        self.instances: List[Instance] = list(instances_set.keys())

    def get_request_states(
        self, train_trial_index: int, instance: Instance, reference_index: Optional[int]
    ) -> List[RequestState]:
        return self.request_state_map.get((train_trial_index, instance, reference_index), [])
