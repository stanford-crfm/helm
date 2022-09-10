from dataclasses import dataclass
from typing import List, Optional
import dacite
import yaml

from common.hierarchical_logger import htrack, hlog
from proxy.models import MODEL_NAME_TO_MODEL
from benchmark.presentation.schema import Schema

CONTAMINATION_YAML_PATH: str = "src/proxy/static/contamination.yaml"

CONTAMINATION_SYMBOLS = {
    "weak": "⚠",
    "strong": "☠",
}

CONTAMINATION_STYLES = {
    "weak": {"color": "gray"},
    "strong": {"color": "lightgray"},
}


@dataclass(frozen=True)
class ContaminationPoint:
    """
    Represents the fact that a set of scenario groups might have been used in
    the training data of some models.
    """

    scenario_groups: List[str]
    models: List[str]
    level: str
    description: str


@dataclass(frozen=True)
class Contamination:
    """
    Captures train-test contamination information between models and benchmark.
    """

    points: List[ContaminationPoint]

    def get_point(self, scenario_group: str, model: str) -> Optional[ContaminationPoint]:
        """Return the point that matches `scenario_group` and `model`."""
        found_points = [
            point for point in self.points if scenario_group in point.scenario_groups and model in point.models
        ]
        assert len(found_points) <= 1
        return found_points[0] if len(found_points) == 1 else None


@htrack(None)
def validate_contamination(contamination: Contamination, schema: Schema):
    """Make sure scenario groups and models in contamination are defined according to `schema`."""
    for point in contamination.points:
        for scenario_group in point.scenario_groups:
            if scenario_group not in schema.name_to_scenario_group:
                hlog(f"WARNING: scenario group {scenario_group} not defined in schema")
        for model in point.models:
            if model not in MODEL_NAME_TO_MODEL:
                hlog(f"WARNING: model {model} not defined in schema")


def read_contamination():
    hlog(f"Reading contamination information from {CONTAMINATION_YAML_PATH}...")
    with open(CONTAMINATION_YAML_PATH) as f:
        raw = yaml.safe_load(f)
    return dacite.from_dict(Contamination, raw)
