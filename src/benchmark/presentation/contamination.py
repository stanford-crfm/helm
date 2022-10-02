from dataclasses import dataclass
from typing import List, Optional
import dacite
import yaml

from common.hierarchical_logger import htrack, hlog
from proxy.models import MODEL_NAME_TO_MODEL
from benchmark.presentation.schema import Schema

CONTAMINATION_YAML_PATH: str = "src/proxy/static/contamination.yaml"

# Contamination levels
CONTAMINATION_LEVEL_WEAK = "weak"
CONTAMINATION_LEVEL_MEDIUM = "medium"
CONTAMINATION_LEVEL_STRONG = "strong"

CONTAMINATION_SYMBOLS = {
    CONTAMINATION_LEVEL_WEAK: "⚠",
    CONTAMINATION_LEVEL_MEDIUM: "💀",
    CONTAMINATION_LEVEL_STRONG: "☠",
}

# These are CSS styles applied to cells that have the type of contamination.
CONTAMINATION_STYLES = {
    CONTAMINATION_LEVEL_WEAK: {"color": "darkgray"},
    CONTAMINATION_LEVEL_MEDIUM: {"color": "gray"},
    CONTAMINATION_LEVEL_STRONG: {"color": "lightgray"},
}


@dataclass(frozen=True)
class ContaminationPoint:
    """
    Represents the fact that each model in `models` might have been trained on
    data in each group in `scenario_groups`.
    Note this implicitly represents |models| x |scenario_groups| points.
    """

    # Which models
    models: List[str]

    scenario_groups: List[str]

    # How contaminated (strong, medium, or weak)
    level: str

    # Explanation of how we know
    description: str


@dataclass(frozen=True)
class Contamination:
    """
    Captures train-test contamination information between models and scenario groups.
    """

    points: List[ContaminationPoint]

    def get_point(self, model: str, scenario_group: str) -> Optional[ContaminationPoint]:
        """Return the point that matches `scenario_group` and `model`."""
        found_points = [
            point for point in self.points if scenario_group in point.scenario_groups and model in point.models
        ]
        # Note: if more than one found, ideally we should take the strongest
        # one, but leaving for now.
        assert len(found_points) <= 1
        return found_points[0] if len(found_points) == 1 else None


@htrack(None)
def validate_contamination(contamination: Contamination, schema: Schema):
    """Make sure models and scenario groups in contamination are defined according to `schema`."""
    for point in contamination.points:
        for model in point.models:
            if model not in MODEL_NAME_TO_MODEL:
                hlog(f"WARNING: model {model} not defined in schema")
        for scenario_group in point.scenario_groups:
            if scenario_group not in schema.name_to_run_group:
                hlog(f"WARNING: scenario group {scenario_group} not defined in schema")


def read_contamination():
    hlog(f"Reading contamination information from {CONTAMINATION_YAML_PATH}...")
    with open(CONTAMINATION_YAML_PATH) as f:
        raw = yaml.safe_load(f)
    return dacite.from_dict(Contamination, raw)
