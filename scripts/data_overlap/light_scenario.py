from dataclasses import dataclass
from typing import List, Optional
from helm.benchmark.scenarios.scenario import ScenarioSpec


@dataclass(frozen=True, eq=False)
class LightInstance:
    """
    A lighter `Instance` with only text fields.
    """

    """The input"""
    input: str

    """References that help us evaluate"""
    references: List[str]

    """Helm instance id"""
    id: Optional[str] = None


@dataclass(frozen=True, eq=False)
class LightScenario:
    """
    A lighter `Scenario`.
    """

    scenario_spec: ScenarioSpec

    split: str

    instances: List[LightInstance]
    """Instances of this scenario"""
