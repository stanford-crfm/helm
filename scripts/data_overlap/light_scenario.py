from dataclasses import dataclass
from typing import List, Dict, Hashable


@dataclass(frozen=True, eq=False)
class LightInstance:
    """
    A lighter `Instance` with only text fields.
    """

    input: str
    """The input"""

    references: List[str]
    """References that help us evaluate"""


@dataclass(frozen=True)
class LightScenarioKey:
    """Unique key representing a `LightScenario` instance."""

    metadata: Dict[str, Hashable]

    def __hash__(self):
        return hash(tuple((k, self.metadata[k]) for k in sorted(self.metadata.keys())))


@dataclass(frozen=True, eq=False)
class LightScenario:
    """
    A lighter `Scenario`.
    """

    light_scenario_key: LightScenarioKey

    light_instances: List[LightInstance]
    """Instances of this scenario"""
