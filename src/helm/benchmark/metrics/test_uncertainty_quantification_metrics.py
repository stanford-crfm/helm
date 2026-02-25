"""
Tests for Uncertainty Quantification Metrics
"""

import math
import pytest

from helm.common.request import Token, GeneratedOutput, Request, RequestResult
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.uncertainty_quantification_metrics import UncertaintyQuantificationMetric
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.scenarios.scenario import Instance, Reference, Input, Output


def create_mock_request_state_with_logprobs(logprobs: list) -> RequestState:
    """Create a mock RequestState with specified log probabilities."""
    tokens = [Token(text=f"token_{i}", logprob=lp) for i, lp in enumerate(logprobs)]
    completion = GeneratedOutput(text="test output", logprob=sum(logprobs), tokens=tokens)
    request = Request(
        model="test_model",
        prompt="test prompt",
        num_completions=1,
        temperature=0.0,
    )
    result = RequestResult(
        success=True,
        completions=[completion],
        embedding=[],
    )
    instance = Instance(
        id="test_instance",
        input=Input(text="test input"),
        references=[Reference(Output(text="test reference"), tags=[])],
    )
    return RequestState(
        instance=instance,
        request=request,
        result=result,
        train_trial_index=0,
        output_mapping=None,
        reference_index=None,
        request_mode=None,
    )


def test_uncertainty_quantification_entropy():
    """Test entropy computation."""
    metric = UncertaintyQuantificationMetric()
    
    # Test with uniform distribution (high entropy)
    logprobs = [-math.log(5)] * 5  # 5 equally likely outcomes
    request_state = create_mock_request_state_with_logprobs(logprobs)
    
    stats = metric.evaluate_generation(
        adapter_spec=None,
        request_state=request_state,
        metric_service=MetricService(),
        eval_cache_path="",
    )
    
    # Find entropy stat
    entropy_stat = next((s for s in stats if s.name.name == "prediction_entropy"), None)
    assert entropy_stat is not None
    assert entropy_stat.mean is not None
    # Should be close to log(5) for uniform distribution
    assert abs(entropy_stat.mean - math.log(5)) < 0.1


def test_uncertainty_quantification_max_probability():
    """Test max probability computation."""
    metric = UncertaintyQuantificationMetric()
    
    # Test with concentrated distribution (one high probability)
    logprobs = [0.0, -10.0, -10.0, -10.0]  # First outcome has high probability
    request_state = create_mock_request_state_with_logprobs(logprobs)
    
    stats = metric.evaluate_generation(
        adapter_spec=None,
        request_state=request_state,
        metric_service=MetricService(),
        eval_cache_path="",
    )
    
    # Find max_probability stat
    max_prob_stat = next((s for s in stats if s.name.name == "max_probability"), None)
    assert max_prob_stat is not None
    assert max_prob_stat.mean is not None
    # Should be close to 1.0 for concentrated distribution
    assert max_prob_stat.mean > 0.9


def test_uncertainty_quantification_coverage():
    """Test coverage metrics."""
    metric = UncertaintyQuantificationMetric(confidence_levels=[0.8, 0.9])
    
    logprobs = [-math.log(10)] * 10  # 10 equally likely outcomes
    request_state = create_mock_request_state_with_logprobs(logprobs)
    
    stats = metric.evaluate_generation(
        adapter_spec=None,
        request_state=request_state,
        metric_service=MetricService(),
        eval_cache_path="",
    )
    
    # Check coverage metrics exist
    coverage_80 = next((s for s in stats if s.name.name == "coverage_at_80"), None)
    coverage_90 = next((s for s in stats if s.name.name == "coverage_at_90"), None)
    assert coverage_80 is not None
    assert coverage_90 is not None


def test_uncertainty_quantification_no_logprobs():
    """Test that metric handles missing logprobs gracefully."""
    metric = UncertaintyQuantificationMetric()
    
    # Create request state without logprobs
    tokens = [Token(text="token", logprob=None) for _ in range(5)]
    completion = GeneratedOutput(text="test", logprob=None, tokens=tokens)
    request = Request(model="test", prompt="test", num_completions=1, temperature=0.0)
    result = RequestResult(success=True, completions=[completion], embedding=[])
    instance = Instance(
        id="test",
        input=Input(text="test"),
        references=[Reference(Output(text="ref"), tags=[])],
    )
    request_state = RequestState(
        instance=instance,
        request=request,
        result=result,
        train_trial_index=0,
        output_mapping=None,
        reference_index=None,
        request_mode=None,
    )
    
    stats = metric.evaluate_generation(
        adapter_spec=None,
        request_state=request_state,
        metric_service=MetricService(),
        eval_cache_path="",
    )
    
    # Should return empty stats when no logprobs available
    assert len(stats) == 0


def test_uncertainty_quantification_metadata():
    """Test that metadata is properly defined."""
    metric = UncertaintyQuantificationMetric()
    metadata = metric.get_metadata()
    
    assert len(metadata) > 0
    # Check that key metrics have metadata
    metric_names = [m.name for m in metadata]
    assert "prediction_entropy" in metric_names
    assert "max_probability" in metric_names
    assert "entropy_ratio" in metric_names

