"""
Tests for Robustness Metrics
"""

import pytest

from helm.common.request import Token, GeneratedOutput, Request, RequestResult
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.adaptation.scenario_state import ScenarioState
from helm.benchmark.metrics.robustness_metrics import RobustnessMetric
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.scenarios.scenario import Instance, Reference, Input, Output


def create_mock_request_state(
    instance_id: str,
    output_text: str,
    train_trial_index: int = 0,
    perturbation_name: str = None,
) -> RequestState:
    """Create a mock RequestState."""
    tokens = [Token(text="token", logprob=-1.0) for _ in range(5)]
    completion = GeneratedOutput(text=output_text, logprob=-5.0, tokens=tokens)
    request = Request(model="test_model", prompt="test prompt", num_completions=1, temperature=0.0)
    result = RequestResult(success=True, completions=[completion], embedding=[])
    
    # Create perturbation if specified
    perturbation = None
    if perturbation_name:
        from helm.benchmark.augmentations.perturbation_description import PerturbationDescription
        perturbation = PerturbationDescription(name=perturbation_name, seed=0)
    
    instance = Instance(
        id=instance_id,
        input=Input(text="test input"),
        references=[Reference(Output(text="correct answer"), tags=["correct"])],
        perturbation=perturbation,
    )
    
    return RequestState(
        instance=instance,
        request=request,
        result=result,
        train_trial_index=train_trial_index,
        output_mapping=None,
        reference_index=None,
        request_mode=None,
    )


def test_robustness_metric_initialization():
    """Test that robustness metric initializes correctly."""
    metric = RobustnessMetric(
        compute_sensitivity=True,
        compute_stability=True,
        compute_adversarial=True,
    )
    assert metric.compute_sensitivity is True
    assert metric.compute_stability is True
    assert metric.compute_adversarial is True


def test_robustness_metric_metadata():
    """Test that metadata is properly defined."""
    metric = RobustnessMetric()
    metadata = metric.get_metadata()
    
    assert len(metadata) > 0
    metric_names = [m.name for m in metadata]
    assert "output_consistency" in metric_names
    assert "perturbation_sensitivity" in metric_names
    assert "adversarial_robustness" in metric_names


def test_robustness_metric_with_single_trial():
    """Test that metric handles single trial gracefully."""
    metric = RobustnessMetric(compute_stability=True)
    
    # Create scenario state with single trial
    instance = Instance(
        id="test",
        input=Input(text="test"),
        references=[Reference(Output(text="ref"), tags=["correct"])],
    )
    request_state = create_mock_request_state("test", "output", train_trial_index=0)
    
    from helm.benchmark.adaptation.scenario_state import ScenarioState
    from helm.benchmark.adaptation.adapter_spec import AdapterSpec
    from helm.benchmark.config_registry import ModelDeployment
    
    adapter_spec = AdapterSpec(
        method="generation",
        model_deployment=ModelDeployment(name="test", url="test"),
        num_train_trials=1,
    )
    
    scenario_state = ScenarioState(
        adapter_spec=adapter_spec,
        instances=[instance],
        request_states=[request_state],
    )
    
    result = metric.evaluate(
        scenario_state=scenario_state,
        metric_service=MetricService(),
        eval_cache_path="",
        parallelism=1,
    )
    
    # Should return valid result even with single trial
    assert result is not None
    assert isinstance(result.aggregated_stats, list)
    assert isinstance(result.per_instance_stats, list)

