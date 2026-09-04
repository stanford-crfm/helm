from typing import Any, Dict, List
from unittest.mock import Mock

import pytest

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.medcalc_bench_metrics import MedCalcBenchMetric
from helm.benchmark.metrics.statistic import Stat
from helm.benchmark.scenarios.scenario import Input, Instance, Output, Reference, CORRECT_TAG
from helm.common.request import Request, RequestResult, GeneratedOutput


def _request_state(prediction: str, reference: str, extra_data: Dict[str, Any]) -> RequestState:
    return RequestState(
        instance=Instance(
            input=Input(text=""),
            references=[Reference(output=Output(text=reference), tags=[CORRECT_TAG])],
            extra_data=extra_data,
        ),
        reference_index=None,
        request_mode=None,
        train_trial_index=0,
        output_mapping=None,
        request=Request(model="openai/text-davinci-002", model_deployment="openai/text-davinci-002"),
        result=RequestResult(
            success=True,
            embedding=[],
            completions=[GeneratedOutput(text=prediction, logprob=0.0, tokens=[])],
            cached=False,
        ),
        num_train_instances=0,
        prompt_truncated=False,
    )


def _accuracy(stats: List[Stat]) -> float:
    (stat,) = stats
    assert stat.name.name == "medcalc_bench_accuracy"
    assert stat.mean is not None
    return stat.mean


@pytest.mark.parametrize("category", ["lab", "dosage", "physical"])
def test_numeric_categories_are_scored_by_range(category: str):
    metric = MedCalcBenchMetric()
    extra_data = {"lower_limit": "287.8766", "upper_limit": "318.1794"}
    assert metric.check_in_range("303", "303.03", extra_data, category) == 1
    assert metric.check_in_range("287.8766", "303.03", extra_data, category) == 1
    assert metric.check_in_range("250", "303.03", extra_data, category) == 0
    assert metric.check_in_range("not a number", "303.03", extra_data, category) == 0


def test_unsupported_category_raises():
    with pytest.raises(ValueError, match="Category unknown not supported"):
        MedCalcBenchMetric().check_in_range("1", "1", {"lower_limit": "0", "upper_limit": "2"}, "unknown")


def test_evaluate_generation_scores_lab_prediction_within_range():
    extra_data = {"category": "lab", "lower_limit": "15.2", "upper_limit": "16.8"}
    request_state = _request_state(prediction="15.4", reference="16.0", extra_data=extra_data)
    stats = MedCalcBenchMetric().evaluate_generation(AdapterSpec(), request_state, Mock(), "")
    assert _accuracy(stats) == 1
