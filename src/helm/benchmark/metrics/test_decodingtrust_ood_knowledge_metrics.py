from typing import Dict, List

import pytest

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.decodingtrust_ood_knowledge_metrics import OODKnowledgeMetric
from helm.benchmark.scenarios.scenario import CORRECT_TAG, Input, Instance, Output, Reference
from helm.common.request import GeneratedOutput, Request, RequestResult


def _request_state(completion: str) -> RequestState:
    instance = Instance(
        input=Input(text="question"),
        references=[Reference(Output(text="paris"), tags=[CORRECT_TAG]), Reference(Output(text="rome"), tags=[])],
    )
    result = RequestResult(
        success=True,
        embedding=[],
        completions=[GeneratedOutput(text=completion, logprob=0.0, tokens=[])],
        cached=False,
    )
    return RequestState(
        instance=instance,
        reference_index=None,
        request_mode=None,
        train_trial_index=0,
        output_mapping={"A": "paris", "B": "rome"},
        request=Request(model="test/model", model_deployment="test/model"),
        result=result,
        num_train_instances=0,
        prompt_truncated=False,
    )


def _evaluate(completions: List[str]) -> Dict[str, float]:
    stats = OODKnowledgeMetric().evaluate_instances([_request_state(c) for c in completions], "")
    return {stat.name.name: stat.mean for stat in stats}


def test_uppercase_options_are_parsed():
    # "A" is correct, "B" is wrong, "E" is the refusal option.
    stats = _evaluate(["A", "B", "E", "A"])
    assert stats["ood_knowledge_acc"] == pytest.approx(0.5)
    assert stats["ood_knowledge_rr"] == pytest.approx(0.25)
    assert stats["ood_knowledge_macc"] == pytest.approx(0.5 / 0.75)


def test_refusal_keywords_without_option():
    stats = _evaluate(["Sorry, I don't know", "A"])
    assert stats["ood_knowledge_rr"] == pytest.approx(0.5)
    assert stats["ood_knowledge_macc"] == pytest.approx(1.0)


def test_all_refusals_do_not_divide_by_zero():
    stats = _evaluate(["E", "E"])
    assert stats["ood_knowledge_rr"] == pytest.approx(1.0)
    assert stats["ood_knowledge_acc"] == 0
    assert stats["ood_knowledge_macc"] == 0
