from collections import OrderedDict, defaultdict
from dataclasses import dataclass
import os
from typing import Dict, Iterable, List, Optional, Set, Tuple, Any

from helm.benchmark.adaptation.adapter_spec import (
    ADAPT_MULTIPLE_CHOICE_SEPARATE_METHODS,
    ADAPT_MULTIPLE_CHOICE_SEPARATE_CALIBRATED,
)
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.adaptation.scenario_state import ScenarioState
from helm.benchmark.augmentations.perturbation_description import PerturbationDescription
from helm.benchmark.metrics.metric import PerInstanceStats
from helm.common.multimodal_request_utils import gather_generated_image_locations
from helm.benchmark.presentation.schema import Schema
from helm.benchmark.run_spec import RunSpec
from helm.benchmark.scenarios.scenario import Instance
from helm.common.general import write
from helm.common.hierarchical_logger import hlog, htrack
from helm.common.images_utils import encode_base64
from helm.common.request import Request
from helm.common.codec import from_json, to_json


@dataclass(frozen=True)
class DisplayPrediction:
    """
    Captures a unit of evaluation for displaying in the web frontend.
    """

    # (instance_id, perturbation, train_trial_index) is a unique key for this prediction.
    instance_id: str
    """ID of the Instance"""

    perturbation: Optional[PerturbationDescription]
    """Description of the Perturbation that was applied"""

    train_trial_index: int
    """Which replication"""

    predicted_text: str
    """Prediction text"""

    truncated_predicted_text: Optional[str]
    """The truncated prediction text, if truncation is required by the Adapter method."""

    base64_images: Optional[List[str]]
    """Images in base64."""

    mapped_output: Optional[str]
    """The mapped output, if an output mapping exists and the prediction can be mapped"""

    reference_index: Optional[int]
    """Which reference of the instance we're evaluating (if any)"""

    stats: Dict[str, float]
    """Statistics computed from the predicted output"""

    annotations: Optional[Dict[str, Any]]


@dataclass(frozen=True)
class DisplayRequest:
    """
    Captures a unit of evaluation for displaying in the web frontend.
    """

    # (instance_id, perturbation, train_trial_index) is a unique key for this prediction.
    instance_id: str
    """ID of the Instance"""

    perturbation: Optional[PerturbationDescription]
    """Description of the Perturbation that was applied"""

    train_trial_index: int
    """Which replication"""

    request: Request
    """The actual Request to display in the web frontend.

    There can be multiple requests per trial. The displayed request should be the
    most relevant request e.g. the request for the chosen choice for multiple choice questions."""


def _read_scenario_state(scenario_state_path: str) -> ScenarioState:
    if not os.path.exists(scenario_state_path):
        raise ValueError(f"Could not load ScenarioState from {scenario_state_path}")
    with open(scenario_state_path) as f:
        return from_json(f.read(), ScenarioState)


def _read_per_instance_stats(per_instance_stats_path: str) -> List[PerInstanceStats]:
    if not os.path.exists(per_instance_stats_path):
        raise ValueError(f"Could not load PerInstanceStats from {per_instance_stats_path}")
    with open(per_instance_stats_path) as f:
        return from_json(f.read(), List[PerInstanceStats])


def _truncate_predicted_text(
    predicted_text: str, request_state: RequestState, adapter_spec: AdapterSpec
) -> Optional[str]:
    method = adapter_spec.method
    prefix = ""
    if method in ADAPT_MULTIPLE_CHOICE_SEPARATE_METHODS:
        prefix = request_state.instance.input.text
    elif method == "language_modeling":
        if request_state.result is not None and request_state.result.completions:
            tokens = request_state.result.completions[0].tokens
            if tokens:
                first_token = tokens[0]
                prefix = first_token.text
    if prefix:
        predicted_text = predicted_text
        prefix = prefix
        if predicted_text.startswith(prefix):
            return predicted_text[len(prefix) :]
    return None


def _get_metric_names_for_group(run_group_name: str, schema: Schema) -> Set[str]:
    metric_groups_by_name = {metric_group.name: metric_group for metric_group in schema.metric_groups}
    run_groups_by_name = {run_group.name: run_group for run_group in schema.run_groups}

    result: Set[str] = set()
    run_group = run_groups_by_name.get(run_group_name)
    if run_group is None:
        return result

    for metric_group_name in run_group.metric_groups:
        metric_group = metric_groups_by_name.get(metric_group_name)
        if metric_group is None:
            continue
        for metric_name_matcher in metric_group.metrics:
            if metric_name_matcher.perturbation_name and metric_name_matcher.perturbation_name != "__all__":
                continue
            result.add(metric_name_matcher.substitute(run_group.environment).name)
    return result


def _get_metric_names_for_groups(run_group_names: Iterable[str], schema: Schema) -> Set[str]:
    result: Set[str] = set()
    for run_group_name in run_group_names:
        result.update(_get_metric_names_for_group(run_group_name, schema))
    return result


_INSTANCES_JSON_FILE_NAME = "instances.json"
_DISPLAY_PREDICTIONS_JSON_FILE_NAME = "display_predictions.json"
_DISPLAY_REQUESTS_JSON_FILE_NAME = "display_requests.json"


@htrack(None)
def write_run_display_json(run_path: str, run_spec: RunSpec, schema: Schema, skip_completed: bool) -> None:
    """Write run JSON files that are used by the web frontend.

    The derived JSON files that are used by the web frontend are much more compact than
    the source JSON files. This speeds up web frontend loading significantly.

    Reads:

    - ScenarioState from `scenario_state.json`
    - List[PerInstanceStats] from `per_instance_stats.json`

    Writes:

    - List[Instance] to `instances.json`
    - List[DisplayPrediction] to `display_predictions.json`
    - List[DisplayRequest] to `display_requests.json`
    """
    instances_file_path = os.path.join(run_path, _INSTANCES_JSON_FILE_NAME)
    display_predictions_file_path = os.path.join(run_path, _DISPLAY_PREDICTIONS_JSON_FILE_NAME)
    display_requests_file_path = os.path.join(run_path, _DISPLAY_REQUESTS_JSON_FILE_NAME)

    scenario_state_path = os.path.join(run_path, "scenario_state.json")
    per_instance_stats_path = os.path.join(run_path, "per_instance_stats.json")

    if (
        skip_completed
        and os.path.exists(instances_file_path)
        and os.path.exists(display_predictions_file_path)
        and os.path.exists(display_requests_file_path)
    ):
        hlog(
            f"Skipping writing display JSON for run {run_spec.name} "
            "because all output display JSON files already exist."
        )
        return
    elif not os.path.exists(scenario_state_path):
        hlog(
            f"Skipping writing display JSON for run {run_spec.name} because "
            f"the scenario state JSON file does not exist at {scenario_state_path}"
        )
        return
    elif not os.path.exists(per_instance_stats_path):
        hlog(
            f"Skipping writing display JSON for run {run_spec.name} because "
            f"the per instance stats JSON file does not exist at {per_instance_stats_path}"
        )
        return

    scenario_state = _read_scenario_state(scenario_state_path)
    per_instance_stats = _read_per_instance_stats(per_instance_stats_path)

    metric_names = _get_metric_names_for_groups(run_spec.groups, schema)

    if run_spec.adapter_spec.method in ADAPT_MULTIPLE_CHOICE_SEPARATE_METHODS:
        metric_names.add("predicted_index")

    stats_by_trial: Dict[Tuple[str, Optional[PerturbationDescription], int], Dict[str, float]] = defaultdict(dict)
    for original_stats in per_instance_stats:
        stats_dict: Dict[str, float] = {
            original_stat.name.name: original_stat.mean
            for original_stat in original_stats.stats
            if original_stat.name.name in metric_names and original_stat.mean is not None
        }

        key = (
            original_stats.instance_id,
            original_stats.perturbation,
            original_stats.train_trial_index,
        )
        stats_by_trial[key].update(stats_dict)

    instance_id_to_instance: Dict[Tuple[str, Optional[PerturbationDescription]], Instance] = OrderedDict()
    predictions: List[DisplayPrediction] = []
    requests: List[DisplayRequest] = []

    for request_state in scenario_state.request_states:
        assert request_state.instance.id is not None
        if request_state.result is None:
            continue

        # For the multiple_choice_separate_calibrated adapter method,
        # only keep the original prediction and discard the calibration prediction.
        if (
            run_spec.adapter_spec.method == ADAPT_MULTIPLE_CHOICE_SEPARATE_CALIBRATED
            and request_state.request_mode == "calibration"
        ):
            continue

        stats_key = (
            request_state.instance.id,
            request_state.instance.perturbation,
            request_state.train_trial_index,
        )
        trial_stats: Dict[str, float] = stats_by_trial[stats_key]
        # For the multiple_choice_separate_* adapter methods,
        # only keep the prediction for the chosen reference and discard the rest.
        if (
            run_spec.adapter_spec.method in ADAPT_MULTIPLE_CHOICE_SEPARATE_METHODS
            and "predicted_index" in trial_stats
            and trial_stats["predicted_index"] != request_state.reference_index
        ):
            continue

        predicted_text = (
            request_state.result.completions[0].text
            if request_state.result is not None and request_state.result.completions
            else ""
        )
        mapped_output = (
            request_state.output_mapping.get(predicted_text.strip()) if request_state.output_mapping else None
        )
        instance_id_to_instance[(request_state.instance.id, request_state.instance.perturbation)] = (
            request_state.instance
        )

        # Process images and include if they exist
        images: List[str] = [
            encode_base64(image_location)
            for image_location in gather_generated_image_locations(request_state.result)
            if os.path.exists(image_location)
        ]

        predictions.append(
            DisplayPrediction(
                instance_id=request_state.instance.id,
                perturbation=request_state.instance.perturbation,
                train_trial_index=request_state.train_trial_index,
                predicted_text=predicted_text,
                truncated_predicted_text=_truncate_predicted_text(predicted_text, request_state, run_spec.adapter_spec),
                base64_images=images,
                mapped_output=mapped_output,
                reference_index=request_state.reference_index,
                stats=trial_stats,
                annotations=request_state.annotations,
            )
        )
        requests.append(
            DisplayRequest(
                instance_id=request_state.instance.id,
                perturbation=request_state.instance.perturbation,
                train_trial_index=request_state.train_trial_index,
                request=request_state.request,
            )
        )
    write(
        instances_file_path,
        to_json(list(instance_id_to_instance.values())),
    )
    write(display_predictions_file_path, to_json(predictions))
    write(
        display_requests_file_path,
        to_json(requests),
    )
