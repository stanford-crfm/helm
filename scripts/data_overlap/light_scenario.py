from dataclasses import dataclass
from typing import List, Optional

try:
    from scenarios.scenario import ScenarioSpec
except Exception:
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
class LightScenarioKey:
    """
    A lighter `Scenario`.
    """

    scenario_spec: ScenarioSpec

    split: str

    def __eq__(self, other):
        return self.split == other.split and self.scenario_spec.class_name == other.scenario_spec.class_name and self.scenario_spec.args == other.scenario_spec.args
    
    def __hash__(self):
        return hash((self.scenario_spec, self.split))


@dataclass(frozen=True, eq=False)
class LightScenario:
    """
    A lighter `Scenario`.
    """

    scenario_key: LightScenarioKey

    """Instances of this scenario"""
    instances: List[LightInstance]
