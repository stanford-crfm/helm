from dataclasses import dataclass
from typing import List, Optional

try:
    from scenarios.scenario import ScenarioSpec
except Exception:
    from helm.benchmark.scenarios.scenario import ScenarioSpec


@dataclass(frozen=True)
class LightInstance:
    """
    A lighter `Instance` with only text fields.
    """

    input: str
    """The input"""

    references: List[str]
    """References that help us evaluate"""

    id: Optional[str] = None
    """Helm instance id"""


@dataclass(frozen=True)
class LightScenarioKey:
    """
    Key for LightScenario
    """

    scenario_spec: ScenarioSpec

    split: str

    def __hash__(self):
        return hash((self.scenario_spec, self.split))


@dataclass(frozen=True)
class LightScenario:
    """
    A lighter `Scenario`.
    """

    scenario_key: LightScenarioKey

    instances: List[LightInstance]
    """Instances of this scenario"""


@dataclass(frozen=True)
class ScenarioSpecInstanceIds:
    """
    Instance ids associated with a scenario
    """

    scenario_spec: ScenarioSpec

    instance_ids: List[str]
