from typing import List
from unittest.mock import MagicMock, patch
import pytest

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import (
    ADAPT_MULTIPLE_CHOICE_SEPARATE_ORIGINAL,
    ADAPT_MULTIPLE_CHOICE_SEPARATE_CALIBRATED,
)
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.basic_metrics import BasicReferenceMetric
from helm.benchmark.metrics.common_metric_specs import (
    get_basic_metric_specs,
    get_basic_reference_metric_specs,
)
from helm.benchmark.metrics.metric import MetricSpec, create_metric
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat
from helm.benchmark.scenarios.scenario import CORRECT_TAG, Input, Instance, Output, Reference
from helm.benchmark.window_services.window_service import WindowService
from helm.common.request import GeneratedOutput, Request, RequestResult, Token


def _get_stat_value(stats: List[Stat], stat_name: str) -> float:
    for stat in stats:
        if stat.name.name == stat_name:
            assert stat.mean is not None
            return stat.mean
    raise ValueError(f"Stat '{stat_name}' not found in stats.")


def _create_request_state(
    instance: Instance,
    reference_index: int,
    reference_text: str,
    logprob: float,
    request_mode: str = "original",
) -> RequestState:
    tokens = [Token(text=f" {reference_text}", logprob=logprob)]
    result = RequestResult(
        success=True,
        embedding=[],
        completions=[
            GeneratedOutput(
                text=reference_text,
                logprob=logprob,
                tokens=tokens,
                finish_reason={"reason": "endoftext"},
            )
        ],
        cached=False,
    )
    request = Request(
        model="openai/text-davinci-002",
        model_deployment="openai/text-davinci-002",
        prompt="",
    )
    return RequestState(
        instance=instance,
        reference_index=reference_index,
        request_mode=request_mode,
        train_trial_index=0,
        output_mapping=None,
        request=request,
        result=result,
        num_train_instances=0,
        prompt_truncated=False,
    )


def _mock_window_service() -> MagicMock:
    mock_ws = MagicMock(spec=WindowService)
    mock_ws.tokenize.side_effect = lambda text: [text]
    mock_ws.get_num_tokens.return_value = 1
    mock_ws.fits_within_context_window.return_value = True
    mock_ws.truncate_from_right.side_effect = lambda prompt: prompt
    return mock_ws


def test_strategy_initialization():
    default_metric = BasicReferenceMetric()
    assert default_metric.tie_breaking_strategy == "first"
    assert repr(default_metric) == "BasicReferenceMetric"

    first_metric = BasicReferenceMetric(tie_breaking_strategy="first")
    assert first_metric.tie_breaking_strategy == "first"
    assert repr(first_metric) == "BasicReferenceMetric"

    binary_metric = BasicReferenceMetric(tie_breaking_strategy="binary")
    assert binary_metric.tie_breaking_strategy == "first"
    assert repr(binary_metric) == "BasicReferenceMetric"

    case_insensitive_metric = BasicReferenceMetric(tie_breaking_strategy=" FIRST ")
    assert case_insensitive_metric.tie_breaking_strategy == "first"

    fractional_metric = BasicReferenceMetric(tie_breaking_strategy="fractional")
    assert fractional_metric.tie_breaking_strategy == "fractional"
    assert repr(fractional_metric) == "BasicReferenceMetric(tie_breaking_strategy='fractional')"

    fractional_upper_metric = BasicReferenceMetric(tie_breaking_strategy=" Fractional ")
    assert fractional_upper_metric.tie_breaking_strategy == "fractional"

    with pytest.raises(ValueError, match="Unknown tie_breaking_strategy"):
        BasicReferenceMetric(tie_breaking_strategy="invalid_strategy")


@patch("helm.benchmark.metrics.basic_metrics.WindowServiceFactory.get_window_service")
@patch("helm.benchmark.metrics.efficiency_metrics.WindowServiceFactory.get_window_service")
def test_single_argmax_correct(mock_eff_ws, mock_bm_ws):
    mock_ws = _mock_window_service()
    mock_bm_ws.return_value = mock_ws
    mock_eff_ws.return_value = mock_ws

    references = [
        Reference(Output(text="Option A"), tags=[CORRECT_TAG]),
        Reference(Output(text="Option B"), tags=[]),
    ]
    instance = Instance(Input(text="Question"), references=references)
    adapter_spec = AdapterSpec(
        method=ADAPT_MULTIPLE_CHOICE_SEPARATE_ORIGINAL,
        model="openai/text-davinci-002",
        model_deployment="openai/text-davinci-002",
    )

    request_states = [
        _create_request_state(instance, reference_index=0, reference_text="Option A", logprob=-0.5),
        _create_request_state(instance, reference_index=1, reference_text="Option B", logprob=-2.0),
    ]

    metric = BasicReferenceMetric()
    stats = metric.evaluate_references(
        adapter_spec=adapter_spec,
        reference_request_states=request_states,
        metric_service=MagicMock(spec=MetricService),
        eval_cache_path="",
    )

    assert _get_stat_value(stats, "exact_match") == 1.0
    assert _get_stat_value(stats, "predicted_index") == 0


@patch("helm.benchmark.metrics.basic_metrics.WindowServiceFactory.get_window_service")
@patch("helm.benchmark.metrics.efficiency_metrics.WindowServiceFactory.get_window_service")
def test_single_argmax_incorrect(mock_eff_ws, mock_bm_ws):
    mock_ws = _mock_window_service()
    mock_bm_ws.return_value = mock_ws
    mock_eff_ws.return_value = mock_ws

    references = [
        Reference(Output(text="Option A"), tags=[]),
        Reference(Output(text="Option B"), tags=[CORRECT_TAG]),
    ]
    instance = Instance(Input(text="Question"), references=references)
    adapter_spec = AdapterSpec(
        method=ADAPT_MULTIPLE_CHOICE_SEPARATE_ORIGINAL,
        model="openai/text-davinci-002",
        model_deployment="openai/text-davinci-002",
    )

    request_states = [
        _create_request_state(instance, reference_index=0, reference_text="Option A", logprob=-0.5),
        _create_request_state(instance, reference_index=1, reference_text="Option B", logprob=-2.0),
    ]

    metric = BasicReferenceMetric()
    stats = metric.evaluate_references(
        adapter_spec=adapter_spec,
        reference_request_states=request_states,
        metric_service=MagicMock(spec=MetricService),
        eval_cache_path="",
    )

    assert _get_stat_value(stats, "exact_match") == 0.0
    assert _get_stat_value(stats, "predicted_index") == 0


@patch("helm.benchmark.metrics.basic_metrics.WindowServiceFactory.get_window_service")
@patch("helm.benchmark.metrics.efficiency_metrics.WindowServiceFactory.get_window_service")
def test_tied_argmax_first_choice_correct(mock_eff_ws, mock_bm_ws):
    mock_ws = _mock_window_service()
    mock_bm_ws.return_value = mock_ws
    mock_eff_ws.return_value = mock_ws

    # References 0 and 1 tie. Reference 0 is correct.
    references = [
        Reference(Output(text="Option A"), tags=[CORRECT_TAG]),
        Reference(Output(text="Option B"), tags=[]),
    ]
    instance = Instance(Input(text="Question"), references=references)
    adapter_spec = AdapterSpec(
        method=ADAPT_MULTIPLE_CHOICE_SEPARATE_ORIGINAL,
        model="openai/text-davinci-002",
        model_deployment="openai/text-davinci-002",
    )

    request_states = [
        _create_request_state(instance, reference_index=0, reference_text="Option A", logprob=-1.0),
        _create_request_state(instance, reference_index=1, reference_text="Option B", logprob=-1.0),
    ]

    # Default / "first": predicted_index is 0 (first in argmax), which matches correct answer -> 1.0
    metric_first = BasicReferenceMetric()
    stats_first = metric_first.evaluate_references(
        adapter_spec=adapter_spec,
        reference_request_states=request_states,
        metric_service=MagicMock(spec=MetricService),
        eval_cache_path="",
    )
    assert _get_stat_value(stats_first, "exact_match") == 1.0
    assert _get_stat_value(stats_first, "predicted_index") == 0

    # "fractional": 1 correct reference out of 2 tied -> 0.5
    metric_frac = BasicReferenceMetric(tie_breaking_strategy="fractional")
    stats_frac = metric_frac.evaluate_references(
        adapter_spec=adapter_spec,
        reference_request_states=request_states,
        metric_service=MagicMock(spec=MetricService),
        eval_cache_path="",
    )
    assert _get_stat_value(stats_frac, "exact_match") == 0.5
    assert _get_stat_value(stats_frac, "predicted_index") == 0


@patch("helm.benchmark.metrics.basic_metrics.WindowServiceFactory.get_window_service")
@patch("helm.benchmark.metrics.efficiency_metrics.WindowServiceFactory.get_window_service")
def test_tied_argmax_second_choice_correct(mock_eff_ws, mock_bm_ws):
    mock_ws = _mock_window_service()
    mock_bm_ws.return_value = mock_ws
    mock_eff_ws.return_value = mock_ws

    # References 0 and 1 tie. Reference 1 is correct.
    references = [
        Reference(Output(text="Option A"), tags=[]),
        Reference(Output(text="Option B"), tags=[CORRECT_TAG]),
    ]
    instance = Instance(Input(text="Question"), references=references)
    adapter_spec = AdapterSpec(
        method=ADAPT_MULTIPLE_CHOICE_SEPARATE_ORIGINAL,
        model="openai/text-davinci-002",
        model_deployment="openai/text-davinci-002",
    )

    request_states = [
        _create_request_state(instance, reference_index=0, reference_text="Option A", logprob=-1.0),
        _create_request_state(instance, reference_index=1, reference_text="Option B", logprob=-1.0),
    ]

    # Default / "first": predicted_index is 0 (first in argmax), which is incorrect -> binary 0.0
    metric_first = BasicReferenceMetric()
    stats_first = metric_first.evaluate_references(
        adapter_spec=adapter_spec,
        reference_request_states=request_states,
        metric_service=MagicMock(spec=MetricService),
        eval_cache_path="",
    )
    assert _get_stat_value(stats_first, "exact_match") == 0.0
    assert _get_stat_value(stats_first, "predicted_index") == 0

    # "fractional": 1 correct reference out of 2 tied -> fractional 0.5
    metric_frac = BasicReferenceMetric(tie_breaking_strategy="fractional")
    stats_frac = metric_frac.evaluate_references(
        adapter_spec=adapter_spec,
        reference_request_states=request_states,
        metric_service=MagicMock(spec=MetricService),
        eval_cache_path="",
    )
    assert _get_stat_value(stats_frac, "exact_match") == 0.5
    assert _get_stat_value(stats_frac, "predicted_index") == 0


@patch("helm.benchmark.metrics.basic_metrics.WindowServiceFactory.get_window_service")
@patch("helm.benchmark.metrics.efficiency_metrics.WindowServiceFactory.get_window_service")
def test_all_tied_four_choices(mock_eff_ws, mock_bm_ws):
    mock_ws = _mock_window_service()
    mock_bm_ws.return_value = mock_ws
    mock_eff_ws.return_value = mock_ws

    references = [
        Reference(Output(text="Option A"), tags=[]),
        Reference(Output(text="Option B"), tags=[]),
        Reference(Output(text="Option C"), tags=[CORRECT_TAG]),
        Reference(Output(text="Option D"), tags=[]),
    ]
    instance = Instance(Input(text="Question"), references=references)
    adapter_spec = AdapterSpec(
        method=ADAPT_MULTIPLE_CHOICE_SEPARATE_ORIGINAL,
        model="openai/text-davinci-002",
        model_deployment="openai/text-davinci-002",
    )

    request_states = [
        _create_request_state(instance, reference_index=i, reference_text=f"Option {chr(65 + i)}", logprob=-2.0)
        for i in range(4)
    ]

    # Default / "first": predicted_index is 0, which does not match C (index 2) -> 0.0
    metric_first = BasicReferenceMetric()
    stats_first = metric_first.evaluate_references(
        adapter_spec=adapter_spec,
        reference_request_states=request_states,
        metric_service=MagicMock(spec=MetricService),
        eval_cache_path="",
    )
    assert _get_stat_value(stats_first, "exact_match") == 0.0
    assert _get_stat_value(stats_first, "predicted_index") == 0

    # "fractional": 1 correct out of 4 tied -> 0.25
    metric_frac = BasicReferenceMetric(tie_breaking_strategy="fractional")
    stats_frac = metric_frac.evaluate_references(
        adapter_spec=adapter_spec,
        reference_request_states=request_states,
        metric_service=MagicMock(spec=MetricService),
        eval_cache_path="",
    )
    assert _get_stat_value(stats_frac, "exact_match") == 0.25
    assert _get_stat_value(stats_frac, "predicted_index") == 0


@patch("helm.benchmark.metrics.basic_metrics.WindowServiceFactory.get_window_service")
@patch("helm.benchmark.metrics.efficiency_metrics.WindowServiceFactory.get_window_service")
def test_tied_argmax_multiple_correct_answers(mock_eff_ws, mock_bm_ws):
    mock_ws = _mock_window_service()
    mock_bm_ws.return_value = mock_ws
    mock_eff_ws.return_value = mock_ws

    # References 0 and 1 tie. Both are marked correct.
    references = [
        Reference(Output(text="Option A"), tags=[CORRECT_TAG]),
        Reference(Output(text="Option B"), tags=[CORRECT_TAG]),
        Reference(Output(text="Option C"), tags=[]),
    ]
    instance = Instance(Input(text="Question"), references=references)
    adapter_spec = AdapterSpec(
        method=ADAPT_MULTIPLE_CHOICE_SEPARATE_ORIGINAL,
        model="openai/text-davinci-002",
        model_deployment="openai/text-davinci-002",
    )

    request_states = [
        _create_request_state(instance, reference_index=0, reference_text="Option A", logprob=-1.0),
        _create_request_state(instance, reference_index=1, reference_text="Option B", logprob=-1.0),
        _create_request_state(instance, reference_index=2, reference_text="Option C", logprob=-3.0),
    ]

    metric_first = BasicReferenceMetric()
    stats_first = metric_first.evaluate_references(
        adapter_spec=adapter_spec,
        reference_request_states=request_states,
        metric_service=MagicMock(spec=MetricService),
        eval_cache_path="",
    )
    assert _get_stat_value(stats_first, "exact_match") == 1.0
    assert _get_stat_value(stats_first, "predicted_index") == 0

    metric_frac = BasicReferenceMetric(tie_breaking_strategy="fractional")
    stats_frac = metric_frac.evaluate_references(
        adapter_spec=adapter_spec,
        reference_request_states=request_states,
        metric_service=MagicMock(spec=MetricService),
        eval_cache_path="",
    )
    # Both tied references 0 and 1 are correct -> 2 / 2 = 1.0
    assert _get_stat_value(stats_frac, "exact_match") == 1.0
    assert _get_stat_value(stats_frac, "predicted_index") == 0


@patch("helm.benchmark.metrics.basic_metrics.WindowServiceFactory.get_window_service")
@patch("helm.benchmark.metrics.efficiency_metrics.WindowServiceFactory.get_window_service")
def test_tied_argmax_partial_multiple_correct(mock_eff_ws, mock_bm_ws):
    mock_ws = _mock_window_service()
    mock_bm_ws.return_value = mock_ws
    mock_eff_ws.return_value = mock_ws

    # 4 references. References 1 and 2 are correct.
    # References 0, 1, 2 tie with highest logprob (-1.0). Reference 3 has lower logprob (-2.0).
    references = [
        Reference(Output(text="Option A"), tags=[]),
        Reference(Output(text="Option B"), tags=[CORRECT_TAG]),
        Reference(Output(text="Option C"), tags=[CORRECT_TAG]),
        Reference(Output(text="Option D"), tags=[]),
    ]
    instance = Instance(Input(text="Question"), references=references)
    adapter_spec = AdapterSpec(
        method=ADAPT_MULTIPLE_CHOICE_SEPARATE_ORIGINAL,
        model="openai/text-davinci-002",
        model_deployment="openai/text-davinci-002",
    )

    request_states = [
        _create_request_state(instance, reference_index=0, reference_text="Option A", logprob=-1.0),
        _create_request_state(instance, reference_index=1, reference_text="Option B", logprob=-1.0),
        _create_request_state(instance, reference_index=2, reference_text="Option C", logprob=-1.0),
        _create_request_state(instance, reference_index=3, reference_text="Option D", logprob=-2.0),
    ]

    # Default / "first": predicted_index is 0 (not in {1, 2}) -> 0.0
    metric_first = BasicReferenceMetric()
    stats_first = metric_first.evaluate_references(
        adapter_spec=adapter_spec,
        reference_request_states=request_states,
        metric_service=MagicMock(spec=MetricService),
        eval_cache_path="",
    )
    assert _get_stat_value(stats_first, "exact_match") == 0.0
    assert _get_stat_value(stats_first, "predicted_index") == 0

    # "fractional": 2 correct references ({1, 2}) out of 3 tied ({0, 1, 2}) -> 2/3
    metric_frac = BasicReferenceMetric(tie_breaking_strategy="fractional")
    stats_frac = metric_frac.evaluate_references(
        adapter_spec=adapter_spec,
        reference_request_states=request_states,
        metric_service=MagicMock(spec=MetricService),
        eval_cache_path="",
    )
    assert _get_stat_value(stats_frac, "exact_match") == pytest.approx(2.0 / 3.0)
    assert _get_stat_value(stats_frac, "predicted_index") == 0


@patch("helm.benchmark.metrics.basic_metrics.WindowServiceFactory.get_window_service")
@patch("helm.benchmark.metrics.efficiency_metrics.WindowServiceFactory.get_window_service")
def test_tied_argmax_no_correct_answers(mock_eff_ws, mock_bm_ws):
    mock_ws = _mock_window_service()
    mock_bm_ws.return_value = mock_ws
    mock_eff_ws.return_value = mock_ws

    # References with NO correct tag
    references = [
        Reference(Output(text="Option A"), tags=[]),
        Reference(Output(text="Option B"), tags=[]),
    ]
    instance = Instance(Input(text="Question"), references=references)
    adapter_spec = AdapterSpec(
        method=ADAPT_MULTIPLE_CHOICE_SEPARATE_ORIGINAL,
        model="openai/text-davinci-002",
        model_deployment="openai/text-davinci-002",
    )

    request_states = [
        _create_request_state(instance, reference_index=0, reference_text="Option A", logprob=-1.0),
        _create_request_state(instance, reference_index=1, reference_text="Option B", logprob=-1.0),
    ]

    metric_first = BasicReferenceMetric()
    stats_first = metric_first.evaluate_references(
        adapter_spec=adapter_spec,
        reference_request_states=request_states,
        metric_service=MagicMock(spec=MetricService),
        eval_cache_path="",
    )
    assert _get_stat_value(stats_first, "exact_match") == 0.0
    assert _get_stat_value(stats_first, "predicted_index") == 0

    metric_frac = BasicReferenceMetric(tie_breaking_strategy="fractional")
    stats_frac = metric_frac.evaluate_references(
        adapter_spec=adapter_spec,
        reference_request_states=request_states,
        metric_service=MagicMock(spec=MetricService),
        eval_cache_path="",
    )
    assert _get_stat_value(stats_frac, "exact_match") == 0.0
    assert _get_stat_value(stats_frac, "predicted_index") == 0


@patch("helm.benchmark.metrics.basic_metrics.WindowServiceFactory.get_window_service")
@patch("helm.benchmark.metrics.efficiency_metrics.WindowServiceFactory.get_window_service")
def test_calibrated_adapter_tie_handling(mock_eff_ws, mock_bm_ws):
    mock_ws = _mock_window_service()
    mock_bm_ws.return_value = mock_ws
    mock_eff_ws.return_value = mock_ws

    references = [
        Reference(Output(text="Option A"), tags=[CORRECT_TAG]),
        Reference(Output(text="Option B"), tags=[]),
    ]
    instance = Instance(Input(text="Question"), references=references)
    adapter_spec = AdapterSpec(
        method=ADAPT_MULTIPLE_CHOICE_SEPARATE_CALIBRATED,
        model="openai/text-davinci-002",
        model_deployment="openai/text-davinci-002",
    )

    # original - calibration = -2.0 - (-1.0) = -1.0 for both -> ties!
    request_states = [
        _create_request_state(
            instance, reference_index=0, reference_text="Option A", logprob=-2.0, request_mode="original"
        ),
        _create_request_state(
            instance, reference_index=0, reference_text="Option A", logprob=-1.0, request_mode="calibration"
        ),
        _create_request_state(
            instance, reference_index=1, reference_text="Option B", logprob=-3.0, request_mode="original"
        ),
        _create_request_state(
            instance, reference_index=1, reference_text="Option B", logprob=-2.0, request_mode="calibration"
        ),
    ]

    metric = BasicReferenceMetric()
    stats = metric.evaluate_references(
        adapter_spec=adapter_spec,
        reference_request_states=request_states,
        metric_service=MagicMock(spec=MetricService),
        eval_cache_path="",
    )

    assert _get_stat_value(stats, "exact_match") == 1.0
    assert _get_stat_value(stats, "predicted_index") == 0


def test_basic_reference_metric_metadata():
    metric = BasicReferenceMetric()
    metadata = metric.get_metadata()
    names = [m.name for m in metadata]
    assert "max_prob" in names
    assert "exact_match" in names
    assert "predicted_index" in names


def test_metric_spec_and_factory_integration():
    # Test default metric spec creation
    specs_default = get_basic_reference_metric_specs()
    assert len(specs_default) == 1
    assert isinstance(specs_default[0], MetricSpec)
    metric_default = create_metric(specs_default[0])
    assert isinstance(metric_default, BasicReferenceMetric)
    assert metric_default.tie_breaking_strategy == "first"

    # Test fractional metric spec creation
    specs_frac = get_basic_reference_metric_specs(tie_breaking_strategy="fractional")
    assert len(specs_frac) == 1
    assert specs_frac[0].args == {"tie_breaking_strategy": "fractional"}
    metric_frac = create_metric(specs_frac[0])
    assert isinstance(metric_frac, BasicReferenceMetric)
    assert metric_frac.tie_breaking_strategy == "fractional"

    # Test propagation in get_basic_metric_specs
    basic_specs = get_basic_metric_specs(["exact_match"], tie_breaking_strategy="fractional")
    ref_specs = [s for s in basic_specs if s.class_name.endswith("BasicReferenceMetric")]
    assert len(ref_specs) == 1
    assert ref_specs[0].args == {"tie_breaking_strategy": "fractional"}
