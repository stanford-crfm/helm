from dataclasses import dataclass
from typing import List, Optional
import dacite
import importlib_resources as resources
import yaml  # type: ignore

from helm.common.hierarchical_logger import htrack, hlog
from helm.proxy.models import MODEL_NAME_TO_MODEL
from helm.benchmark.presentation.schema import Schema


EFFICIENCY_JSON_PACKAGE: str = "helm.benchmark.static"
EFFICIENCY_JSON_FILENAME: str = "efficiency.yaml"


@dataclass(frozen=True)
class EfficiencyPoint:
    """
    Represents efficiency training information for a model.
    """

    # Which model
    model: str

    # Explanation of how we know
    description: str


@dataclass(frozen=True)
class Efficiency:
    """
    Captures efficiency training information for a model.
    """

    points: List[EfficiencyPoint]

    def get_point(self, model: str, group: str) -> Optional[EfficiencyPoint]:
        """Return the point that matches `group` and `model`."""
        found_points = [point for point in self.points if group == 'efficiency' and model == point.model]
        assert len(found_points) <= 1
        return found_points[0] if len(found_points) == 1 else None


@htrack(None)
def validate_efficiency(efficiency: Efficiency, schema: Schema):
    """Make sure models and groups in efficiency are defined according to `schema`."""
    for point in efficiency.points:
        for model in point.models:
            if model not in MODEL_NAME_TO_MODEL:
                hlog(f"WARNING: model {model} not defined in schema")
        for group in point.groups:
            if group not in schema.name_to_run_group:
                hlog(f"WARNING: group {group} not defined in schema")


def read_efficiency():
    hlog(f"Reading efficiency information from {EFFICIENCY_JSON_FILENAME}...")
    efficiency_path = resources.files(EFFICIENCY_JSON_PACKAGE).joinpath(EFFICIENCY_JSON_FILENAME)
    with efficiency_path.open("r") as f:
        raw = yaml.safe_load(f)
    return dacite.from_dict(Efficiency, raw)
