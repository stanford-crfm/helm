from dataclasses import dataclass
from typing import List
from helm.benchmark.scenarios.scenario import ScenarioSpec

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
    scenario_spec: ScenarioSpec

    split: str


@dataclass(frozen=True)
class LightScenario:
    """
    A lighter `Scenario`.
    """
    scenario_key: LightScenarioKey

    """Instances of this scenario"""
    instances: List[LightInstance]
