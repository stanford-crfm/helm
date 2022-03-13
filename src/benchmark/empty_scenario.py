from typing import List

from .scenario import Scenario, Instance


class EmptyScenario(Scenario):
    """
    Scenario with no Instances.
    You would use this Scenario during interaction mode.
    """

    name: str = "empty"
    description: str = "An empty scenario."
    tags: List[str] = []

    def __init__(self):
        pass

    def get_instances(self) -> List[Instance]:
        """There are no `Instance`s in EmptyScenario."""
        return []
