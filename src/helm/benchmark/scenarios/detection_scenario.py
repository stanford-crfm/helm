from typing import List

from .scenario import Scenario, Instance, Input, TEST_SPLIT


class DetectionScenario(Scenario):
    """
    TODO: include description of the scenario. Link relevant papers.
    See https://github.com/stanford-crfm/helm/blob/vhelm/src/helm/benchmark/scenarios/draw_bench_scenario.py#L11
    as an example.
    """

    name = "detection"
    description = "TODO: include a short description of this scenario."
    tags = ["text-to-image"]

    def __init__(self):
        super().__init__()
        # TODO: initialize instance variables here if necessary. Otherwise, just remove this method.

    def get_instances(self) -> List[Instance]:
        instances: List[Instance] = []

        # TODO: an Instance represents one prompt. See examples of how `get_instances` are implemented:
        # https://github.com/stanford-crfm/helm/blob/vhelm/src/helm/benchmark/scenarios/logos_scenario.py
        # https://github.com/stanford-crfm/helm/blob/vhelm/src/helm/benchmark/scenarios/draw_bench_scenario.py
        # TODO: just an example. delete me later
        instances.append(Instance(Input(text="some prompt"), references=[], split=TEST_SPLIT))

        return instances
