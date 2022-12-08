from collections import defaultdict
from dataclasses import dataclass
import os
import json
from typing import Any, Dict, List, Optional, Set, Tuple, cast

import dacite

from helm.benchmark.adapter import AdapterSpec, RequestState, ScenarioState
from helm.benchmark.augmentations.dialect_perturbation import DialectPerturbation
from helm.benchmark.augmentations.extra_space_perturbation import ExtraSpacePerturbation
from helm.benchmark.augmentations.filler_words_perturbation import FillerWordsPerturbation
from helm.benchmark.augmentations.gender_perturbation import GenderPerturbation
from helm.benchmark.augmentations.misspelling_perturbation import MisspellingPerturbation
from helm.benchmark.augmentations.person_name_perturbation import PersonNamePerturbation
from helm.benchmark.augmentations.perturbation_description import PerturbationDescription
from helm.benchmark.augmentations.space_perturbation import SpacePerturbation
from helm.benchmark.augmentations.synonym_perturbation import SynonymPerturbation
from helm.benchmark.augmentations.typos_perturbation import TyposPerturbation
from helm.benchmark.metrics.metric import PerInstanceStats
from helm.benchmark.presentation.schema import Schema
from helm.benchmark.runner import RunSpec
from helm.common.general import write, asdict_without_nones
from helm.common.hierarchical_logger import htrack
from helm.common.request import Request


_PERTURBATION_NAME_TO_DESCRIPTION = {
    DialectPerturbation.name: DialectPerturbation.Description,
    ExtraSpacePerturbation.name: ExtraSpacePerturbation.Description,
    FillerWordsPerturbation.name: FillerWordsPerturbation.Description,
    GenderPerturbation.name: GenderPerturbation.Description,
    MisspellingPerturbation.name: MisspellingPerturbation.Description,
    PersonNamePerturbation.name: PersonNamePerturbation.Description,
    SpacePerturbation.name: SpacePerturbation.Description,
    SynonymPerturbation.name: SynonymPerturbation.Description,
    TyposPerturbation.name: TyposPerturbation.Description,
}


def _deserialize_perturbation_description(raw_perturbation_description: Dict[Any, Any]) -> PerturbationDescription:
    """Convert a raw dictionary to a PerturbationDescription.
    This uses the name field to look up the correct PerturbationDescription subclass to output.
    """
    # This is brittle because it it will fail on new PerturbationDescription subclasses
    # that don't get added to _PERTURBATION_NAME_TO_DESCRIPTION
    # TODO: Move this to a common utilities library, and add proper class registration
    factory = _PERTURBATION_NAME_TO_DESCRIPTION.get(raw_perturbation_description["name"], PerturbationDescription)
    return factory(**raw_perturbation_description)


_DACITE_CONFIG = dacite.Config(type_hooks={PerturbationDescription: _deserialize_perturbation_description})


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

    mapped_output: Optional[str]
    """The mapped output, if an output mapping exists and the prediction can be mapped"""

    reference_index: Optional[int]
    """Which reference of the instance we're evaluating (if any)"""

    stats: Dict[str, float]
    """Statistics computed from the predicted output"""


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
    most relevant request e.g. the request for the chosen cohice for multiple choice questions."""


def _read_scenario_state(run_path: str) -> ScenarioState:
    scenario_state_path: str = os.path.join(run_path, "scenario_state.json")
    if not os.path.exists(scenario_state_path):
        raise ValueError(f"Could not load ScenarioState from {scenario_state_path}")
    with open(scenario_state_path) as f:
        raw_scenario_state = json.load(f)
        return dacite.from_dict(ScenarioState, raw_scenario_state, config=_DACITE_CONFIG)


def _read_per_instance_stats(run_path: str) -> List[PerInstanceStats]:
    per_instance_stats_path: str = os.path.join(run_path, "per_instance_stats.json")
    if not os.path.exists(per_instance_stats_path):
        raise ValueError(f"Could not load PerInstanceStats from {per_instance_stats_path}")
    with open(per_instance_stats_path) as f:
        raw_per_instance_stats = json.load(f)
        return [dacite.from_dict(PerInstanceStats, r, config=_DACITE_CONFIG) for r in raw_per_instance_stats]


def _truncate_predicted_text(
    predicted_text: str, request_state: RequestState, adapter_spec: AdapterSpec
) -> Optional[str]:
    method = adapter_spec.method
    prefix = ""
    if method.startswith("multiple_choice_separate_"):
        prefix = request_state.instance.input
    elif method == "language_modeling":
        if request_state.result is not None and request_state.result.completions:
            tokens = request_state.result.completions[0].tokens
            if tokens:
                first_token = tokens[0]
                if not first_token.top_logprobs:
                    prefix = first_token.text
    if prefix:
        predicted_text = predicted_text.strip()
        prefix = prefix.strip()
        if predicted_text.startswith(prefix):
            return predicted_text[len(prefix) :].strip()
    return None


@htrack(None)
def write_run_display_json(run_path: str, run_spec: RunSpec, schema: Schema):
    """Write run JSON files that are used by the web frontend.

    The derived JSON files that are used by the web frontend are much more compact than
    the source JSON files. This speeds up web frontend loading significantly.

    Reads:

    - ScenarioState from `scenario_state.json`
    - List[PerInstanceStats] from `per_instance_stats.json`

    Writes:

    - List[Instance] to `instances.json`
    - List[DisplayPrediction] to `predictions.json`
    - List[DisplayRequest] to `requests.json`
    """
    scenario_state = _read_scenario_state(run_path)
    per_instance_stats = _read_per_instance_stats(run_path)

    metric_groups_by_name = {metric_group.name: metric_group for metric_group in schema.metric_groups}
    run_groups_by_name = {run_group.name: run_group for run_group in schema.run_groups}

    metric_names_from_schema: Set[str] = set(
        metric_name_matcher.substitute(run_groups_by_name[run_group_name].environment).name
        for run_group_name in run_spec.groups
        if run_group_name in run_groups_by_name
        for metric_group_name in run_groups_by_name[run_group_name].metric_groups
        if metric_group_name in metric_groups_by_name
        for metric_name_matcher in metric_groups_by_name[metric_group_name].metrics
        if not metric_name_matcher.perturbation_name
    )
    if run_spec.adapter_spec.method.startswith("multiple_choice_separate_"):
        metric_names_from_schema.add("predicted_index")

    stats_by_trial: Dict[Tuple[str, Optional[PerturbationDescription], int], Dict[str, float]] = defaultdict(dict)
    for original_stats in per_instance_stats:
        stats_dict: Dict[str, float] = {
            original_stat.name.name: cast(float, original_stat.mean)
            for original_stat in original_stats.stats
            if original_stat.name.name in metric_names_from_schema
        }
        key = (original_stats.instance_id, original_stats.perturbation, original_stats.train_trial_index)
        stats_by_trial[key].update(stats_dict)

    predictions = []
    requests = []
    for request_state in scenario_state.request_states:
        assert request_state.instance.id is not None
        if request_state.result is None:
            continue

        # For the multiple_choice_separate_calibrated adapter method,
        # only keep the original prediction and discard the calibration prediction.
        if (
            run_spec.adapter_spec.method.startswith("multiple_choice_separate_calibrated")
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
            run_spec.adapter_spec.method.startswith("multiple_choice_separate_")
            and "predicted_index" in trial_stats
            and trial_stats["predicted_index"] != request_state.reference_index
        ):
            continue

        predicted_text = (
            request_state.result.completions[0].text
            if request_state.result is not None or request_state.result.completions
            else None
        )
        mapped_output = (
            request_state.output_mapping.get(predicted_text.strip()) if request_state.output_mapping else None
        )

        predictions.append(
            DisplayPrediction(
                instance_id=request_state.instance.id,
                perturbation=request_state.instance.perturbation,
                train_trial_index=request_state.train_trial_index,
                predicted_text=predicted_text,
                truncated_predicted_text=_truncate_predicted_text(predicted_text, request_state, run_spec.adapter_spec),
                mapped_output=mapped_output,
                reference_index=request_state.reference_index,
                stats=trial_stats,
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
        os.path.join(run_path, "instances.json"),
        json.dumps([asdict_without_nones(instance) for instance in scenario_state.instances], indent=2),
    )
    write(
        os.path.join(run_path, "predictions.json"),
        json.dumps(list(map(asdict_without_nones, predictions)), indent=2),
    )
    write(
        os.path.join(run_path, "requests.json"),
        json.dumps(list(map(asdict_without_nones, requests)), indent=2),
    )
