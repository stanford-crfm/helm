from typing import List

from scenario import create_scenario
from test_utils import get_scenario_spec1, get_adapter_spec1
from adapter import Adapter


def test_adapter1():
    scenario = create_scenario(get_scenario_spec1())
    adapter_spec = get_adapter_spec1()

    scenario_state = Adapter(adapter_spec).adapt(scenario)

    # Make sure we generated the right number of request_states:
    # For each trial, instance and reference (+ 1 for free-form generation).
    num_instances = len(scenario_state.instances)
    num_references = len(scenario_state.instances[0].references)
    assert num_instances * (num_references + 1) * adapter_spec.num_train_trials == len(scenario_state.request_states)
