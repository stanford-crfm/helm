from dataclasses import dataclass
from typing import List, Optional

try:
    from scenarios.scenario import ScenarioSpec
except Exception:
    from helm.benchmark.scenarios.scenario import ScenarioSpec


@dataclass(frozen=True)
class GroupScenarioSpecs:
    """
    Scenario Specs associated with a given Group
    e.g.
    {
        "group": "disinformation_wedging",
        "scenario_specs":
        [
            {
                "class_name": "helm.benchmark.scenarios.disinformation_scenario.DisinformationScenario",
                "args": {"capability": "wedging", "topic": "covid"}
            }
        ]
    }

    """

    group: str

    scenario_specs: List[ScenarioSpec]


@dataclass(frozen=True)
class GroupOverlapStats:
    """
    Dataclass that represents group data overlap stats
    e.g.
    {
        "group": "natural_qa_closedbook",
        "num_instances": 2144,
        "num_overlapping_inputs": 1,
        "num_overlapping_references": 100
    }
    """

    group: str

    num_instances: int

    num_overlapping_inputs: int

    num_overlapping_references: int


@dataclass(frozen=True)
class AllGroupOverlapStats:
    """
    Dataclass that represents all group data overlap stats
    e.g.
    {"models": ["together/bloom", "together/gpt-j-6b", ...],
    "group_overlap_stats_list": [
        {
            "group": "natural_qa_closedbook",
            "num_instances": 2144,
            "num_overlapping_inputs": 1,
            "num_overlapping_references": 100
        }
        ...

    """

    models: List[str]

    group_overlap_stats_list: List[GroupOverlapStats]


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
