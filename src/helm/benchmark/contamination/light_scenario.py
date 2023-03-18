from dataclasses import dataclass
from typing import List


@dataclass(frozen=True, eq=False)
class LightInstance:
    """
    A lighter `Instance` with only text fields.
    """

    input: str
    """The input"""

    references: List[str]
    """References that help us evaluate"""


@dataclass(frozen=True, eq=False)
class LightScenario:
    """
    A lighter `Scenario`.
    """

    scenario_spec: str
    """The scenario spec"""

    light_instances: List[LightInstance]
    """Instances of this scenario"""
