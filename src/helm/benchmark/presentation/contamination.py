from dataclasses import dataclass
from typing import List, Optional
import dacite
import importlib_resources as resources
import yaml

from helm.common.hierarchical_logger import htrack, hlog
from helm.benchmark.model_metadata_registry import MODEL_NAME_TO_MODEL_METADATA
from helm.benchmark.presentation.schema import Schema


CONTAMINATION_YAML_PACKAGE: str = "helm.benchmark.static"
CONTAMINATION_YAML_FILENAME: str = "contamination.yaml"

# Contamination levels
CONTAMINATION_LEVEL_WEAK = "weak"
CONTAMINATION_LEVEL_STRONG = "strong"

CONTAMINATION_SYMBOLS = {
    CONTAMINATION_LEVEL_WEAK: "⚠",
    CONTAMINATION_LEVEL_STRONG: "☠",
}

# These are CSS styles applied to cells that have the type of contamination.
CONTAMINATION_STYLES = {
    CONTAMINATION_LEVEL_WEAK: {"color": "gray"},
    CONTAMINATION_LEVEL_STRONG: {"color": "lightgray"},
}


@dataclass(frozen=True)
class ContaminationPoint:
    """
    Represents the fact that each model in `models` might have been trained on
    data in each group in `groups`.
    Note this implicitly represents |models| x |groups| points.
    """

    # Which models
    models: List[str]

    groups: List[str]

    # How contaminated (strong or weak)
    level: str

    # Explanation of how we know
    description: str


@dataclass(frozen=True)
class Contamination:
    """
    Captures train-test contamination information between models and groups.
    """

    points: List[ContaminationPoint]

    def get_point(self, model: str, group: str) -> Optional[ContaminationPoint]:
        """Return the point that matches `group` and `model`."""
        found_points = [point for point in self.points if group in point.groups and model in point.models]
        # Note: if more than one found, ideally we should take the strongest
        # one, but leaving for now.
        assert len(found_points) <= 1
        return found_points[0] if len(found_points) == 1 else None


@htrack(None)
def validate_contamination(contamination: Contamination, schema: Schema):
    """Make sure models and groups in contamination are defined according to `schema`."""
    for point in contamination.points:
        for model in point.models:
            if model not in MODEL_NAME_TO_MODEL_METADATA:
                hlog(f"WARNING: model {model} not defined in schema")
        for group in point.groups:
            if group not in schema.name_to_run_group:
                hlog(f"WARNING: group {group} not defined in schema")


def read_contamination():
    hlog(f"Reading contamination information from {CONTAMINATION_YAML_FILENAME}...")
    contamination_path = resources.files(CONTAMINATION_YAML_PACKAGE).joinpath(CONTAMINATION_YAML_FILENAME)
    with contamination_path.open("r") as f:
        raw = yaml.safe_load(f)
    return dacite.from_dict(Contamination, raw)
