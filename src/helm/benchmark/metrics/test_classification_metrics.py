from collections import defaultdict
from typing import Dict, List, NamedTuple

from pytest import approx

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.classification_metrics import ClassificationMetric
from helm.benchmark.metrics.statistic import Stat
from helm.benchmark.scenarios.scenario import Input, Instance, Output, Reference, CORRECT_TAG
from helm.common.request import Request, RequestResult, GeneratedOutput


class _Option(NamedTuple):
    text: str
    is_correct: bool


def _request_state(prediction: str, options: List[_Option]):
    references = [
        Reference(output=Output(text=option.text), tags=[CORRECT_TAG] if option.is_correct else [])
        for option in options
    ]
    return RequestState(
        instance=Instance(input=Input(text=""), references=references),
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


def assert_stats_equal(actual_stats: List[Stat], expected_values: Dict[str, float]):
    actual_values = {stat.name.name: stat.mean for stat in actual_stats}
    assert actual_values == approx(expected_values)


def _expected_stats(all_classes_counts: Dict[str, Dict[str, int]]):
    micro_counts: Dict[str, int] = defaultdict(int)
    for class_counts in all_classes_counts.values():
        for key, class_count in class_counts.items():
            micro_counts[key] += class_count
    micro_precision = micro_counts["tp"] / (micro_counts["tp"] + micro_counts["fp"])
    micro_recall = micro_counts["tp"] / (micro_counts["tp"] + micro_counts["fn"])
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall)

    class_f1: List[float] = []
    for class_counts in all_classes_counts.values():
        class_precision = class_counts["tp"] / (class_counts["tp"] + class_counts["fp"])
        class_recall = class_counts["tp"] / (class_counts["tp"] + class_counts["fn"])
        class_f1.append(2 * (class_precision * class_recall) / (class_precision + class_recall))
    macro_f1 = sum(class_f1) / len(class_f1)

    return {
        "classification_micro_f1": micro_f1,
        "classification_macro_f1": macro_f1,
    }


def test_evaluate_instances_binary_generation():
    metric = ClassificationMetric(delimiter=None)

    request_states = [
        _request_state("yes", [_Option("yes", True)]),
        _request_state("yes", [_Option("yes", True)]),
        _request_state("yes", [_Option("yes", True)]),
        _request_state("yes", [_Option("no", True)]),
        _request_state("no", [_Option("yes", True)]),
        _request_state("no", [_Option("no", True)]),
        _request_state("invalid", [_Option("no", True)]),
    ]

    assert_stats_equal(
        metric.evaluate_instances(request_states, ""),
        _expected_stats(
            {
                "yes": {"tp": 3, "fp": 1, "tn": 2, "fn": 1},
                "no": {"tp": 1, "fp": 1, "tn": 3, "fn": 2},
            }
        ),
    )


def test_evaluate_instances_multi_class():
    # Note: no "a" because it would get filtered out by normalize_text()
    metric = ClassificationMetric(delimiter=None)

    def _options(correct: str):
        return [_Option(text, text == correct) for text in ["d", "b", "c"]]

    request_states = [
        _request_state("d", _options("d")),
        _request_state("d", _options("d")),
        _request_state("d", _options("d")),
        _request_state("d", _options("b")),
        _request_state("b", _options("b")),
        _request_state("b", _options("b")),
        _request_state("b", _options("c")),
        _request_state("c", _options("d")),
        _request_state("c", _options("c")),
        _request_state("invalid", _options("c")),
    ]
    assert_stats_equal(
        metric.evaluate_instances(request_states, ""),
        _expected_stats(
            {
                "d": {"tp": 3, "fp": 1, "tn": 5, "fn": 1},
                "b": {"tp": 2, "fp": 1, "tn": 6, "fn": 1},
                "c": {"tp": 1, "fp": 1, "tn": 6, "fn": 2},
            }
        ),
    )


def test_evaluate_instances_multilabel():
    # Note: no "a" because it would get filtered out by normalize_text()
    metric = ClassificationMetric(delimiter=",")

    def _options(correct: List[str]):
        return [_Option(text, text in correct) for text in ["d", "b", "c"]]

    request_states = [
        _request_state("d,b", _options(["d", "b"])),
        _request_state("d,b", _options(["d", "c"])),
        _request_state("d", _options(["d"])),
        _request_state("c", _options(["b"])),
        _request_state("b", _options(["b", "c"])),
        _request_state("d,b", _options(["c"])),
        _request_state("d,c", _options(["d"])),
        _request_state("d,b,c", _options(["d", "b", "c"])),
        _request_state("", []),
        _request_state("n/a", []),
        _request_state("invalid", _options(["c"])),
    ]

    assert_stats_equal(
        metric.evaluate_instances(request_states, ""),
        _expected_stats(
            {
                "d": {"tp": 5, "fp": 1, "tn": 5, "fn": 0},
                "b": {"tp": 3, "fp": 2, "tn": 5, "fn": 1},
                "c": {"tp": 1, "fp": 2, "tn": 4, "fn": 4},
            }
        ),
    )
