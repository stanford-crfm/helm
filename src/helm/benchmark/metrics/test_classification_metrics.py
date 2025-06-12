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


def get_stat_value(stats: List[Stat], stat_name: str):
    for stat in stats:
        if stat.name.name == stat_name:
            return stat.mean
    raise ValueError(f"No stat with name {stat_name}")


def compute_stats(all_classes_counts: Dict[str, Dict[str, int]]):
    micro_counts: Dict[str, int] = defaultdict(int)
    for class_counts in all_classes_counts.values():
        for key, class_count in class_counts.items():
            micro_counts[key] += class_count
    micro_precision = micro_counts["tp"] / (micro_counts["tp"] + micro_counts["fp"])
    micro_recall = micro_counts["tp"] / (micro_counts["tp"] + micro_counts["fn"])
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall)

    class_f1: Dict[str, float] = {}
    for class_name, class_counts in all_classes_counts.items():
        class_precision = class_counts["tp"] / (class_counts["tp"] + class_counts["fp"])
        class_recall = class_counts["tp"] / (class_counts["tp"] + class_counts["fn"])
        class_f1[class_name] = 2 * (class_precision * class_recall) / (class_precision + class_recall)
    macro_f1 = sum(class_f1.values()) / len(class_f1)
    class_name_to_support = {
        class_name: class_counts["tp"] + class_counts["fn"] for class_name, class_counts in all_classes_counts.items()
    }
    weighted_f1 = sum(class_f1[class_name] * support for class_name, support in class_name_to_support.items()) / sum(
        support for support in class_name_to_support.values()
    )

    stats = {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "weighted_f1": weighted_f1,
    }
    for class_name, class_f1_score in class_f1.items():
        stats[f"{class_name}_f1"] = class_f1_score
    return stats


def test_evaluate_instances_default_parameters():
    request_states = [
        _request_state("yes", [_Option("yes", True)]),
        _request_state("yes ", [_Option("yes", True)]),
        _request_state("yeS", [_Option("yes", True)]),
        _request_state("yes", [_Option("no", True)]),
        _request_state("no", [_Option("yes", True)]),
        _request_state("no", [_Option("no", True)]),
        _request_state("invalid", [_Option("no", True)]),
    ]

    expected_stats = compute_stats(
        {
            "yes": {"tp": 3, "fp": 1, "tn": 2, "fn": 1},
            "no": {"tp": 1, "fp": 1, "tn": 3, "fn": 2},
        }
    )

    actual_stats = ClassificationMetric().evaluate_instances(request_states, "")
    actual_macro_f1 = get_stat_value(actual_stats, "classification_macro_f1")
    assert actual_macro_f1 == approx(expected_stats["macro_f1"])
    actual_micro_f1 = get_stat_value(actual_stats, "classification_micro_f1")
    assert actual_micro_f1 == approx(expected_stats["micro_f1"])


def test_evaluate_instances_yes_and_no():
    labels = ["yes", "no"]
    request_states = [
        _request_state("yes", [_Option("yes", True)]),
        _request_state("yes ", [_Option("yes", True)]),
        _request_state("yeS", [_Option("yes", True)]),
        _request_state("yes", [_Option("no", True)]),
        _request_state("no", [_Option("yes", True)]),
        _request_state("no", [_Option("no", True)]),
        _request_state("invalid", [_Option("no", True)]),
    ]

    expected_stats = compute_stats(
        {
            "yes": {"tp": 3, "fp": 1, "tn": 2, "fn": 1},
            "no": {"tp": 1, "fp": 1, "tn": 3, "fn": 2},
        }
    )

    actual_stats = ClassificationMetric(
        scores=["f1"], averages=["macro", "micro", "weighted", None], labels=labels
    ).evaluate_instances(request_states, "")
    actual_macro_f1 = get_stat_value(actual_stats, "classification_macro_f1")
    assert actual_macro_f1 == approx(expected_stats["macro_f1"])
    actual_micro_f1 = get_stat_value(actual_stats, "classification_micro_f1")
    assert actual_micro_f1 == approx(expected_stats["micro_f1"])
    actual_weighted_f1 = get_stat_value(actual_stats, "classification_weighted_f1")
    assert actual_weighted_f1 == approx(expected_stats["weighted_f1"])
    actual_yes_f1 = get_stat_value(actual_stats, "classification_yes_f1")
    assert actual_yes_f1 == approx(expected_stats["yes_f1"])
    actual_no_f1 = get_stat_value(actual_stats, "classification_no_f1")
    assert actual_no_f1 == approx(expected_stats["no_f1"])


def test_evaluate_instances_multi_class():
    labels = ["a", "b", "c"]

    def _gold_label(correct: str):
        return [_Option(text, text == correct) for text in labels]

    request_states = [
        _request_state("a", _gold_label("a")),
        _request_state("a", _gold_label("a")),
        _request_state("a", _gold_label("a")),
        _request_state("a", _gold_label("b")),
        _request_state("b", _gold_label("b")),
        _request_state("b", _gold_label("b")),
        _request_state("b", _gold_label("c")),
        _request_state("c", _gold_label("a")),
        _request_state("c", _gold_label("c")),
        _request_state("invalid", _gold_label("c")),
    ]

    expected_stats = compute_stats(
        {
            "a": {"tp": 3, "fp": 1, "tn": 5, "fn": 1},
            "b": {"tp": 2, "fp": 1, "tn": 6, "fn": 1},
            "c": {"tp": 1, "fp": 1, "tn": 6, "fn": 2},
        }
    )

    actual_stats = ClassificationMetric(
        scores=["f1"], averages=["macro", "micro", "weighted", None], labels=labels
    ).evaluate_instances(request_states, "")
    actual_macro_f1 = get_stat_value(actual_stats, "classification_macro_f1")
    assert actual_macro_f1 == approx(expected_stats["macro_f1"])
    actual_micro_f1 = get_stat_value(actual_stats, "classification_micro_f1")
    assert actual_micro_f1 == approx(expected_stats["micro_f1"])
    actual_weighted_f1 = get_stat_value(actual_stats, "classification_weighted_f1")
    assert actual_weighted_f1 == approx(expected_stats["weighted_f1"])
    actual_a_f1 = get_stat_value(actual_stats, "classification_a_f1")
    assert actual_a_f1 == approx(expected_stats["a_f1"])
    actual_b_f1 = get_stat_value(actual_stats, "classification_b_f1")
    assert actual_b_f1 == approx(expected_stats["b_f1"])
    actual_c_f1 = get_stat_value(actual_stats, "classification_c_f1")
    assert actual_c_f1 == approx(expected_stats["c_f1"])


def test_evaluate_instances_multilabel():
    labels = ["a", "b", "c"]

    def _gold_labels(correct: List[str]):
        return [_Option(text, text in correct) for text in labels]

    request_states = [
        _request_state("a,b", _gold_labels(["a", "b"])),
        _request_state("a,b", _gold_labels(["a", "c"])),
        _request_state("a", _gold_labels(["a"])),
        _request_state("c", _gold_labels(["b"])),
        _request_state("b", _gold_labels(["b", "c"])),
        _request_state("a,b", _gold_labels(["c"])),
        _request_state("a,c", _gold_labels(["a"])),
        _request_state("a,b,c", _gold_labels(["a", "b", "c"])),
        _request_state("", []),
        _request_state("n/a", []),
        _request_state("invalid", _gold_labels(["c"])),
    ]

    expected_stats = compute_stats(
        {
            "a": {"tp": 5, "fp": 1, "tn": 5, "fn": 0},
            "b": {"tp": 3, "fp": 2, "tn": 5, "fn": 1},
            "c": {"tp": 1, "fp": 2, "tn": 4, "fn": 4},
        }
    )

    actual_stats = ClassificationMetric(
        scores=["f1"], averages=["macro", "micro", "weighted", None], labels=labels, delimiter=","
    ).evaluate_instances(request_states, "")
    actual_macro_f1 = get_stat_value(actual_stats, "classification_macro_f1")
    assert actual_macro_f1 == approx(expected_stats["macro_f1"])
    actual_micro_f1 = get_stat_value(actual_stats, "classification_micro_f1")
    assert actual_micro_f1 == approx(expected_stats["micro_f1"])
    actual_weighted_f1 = get_stat_value(actual_stats, "classification_weighted_f1")
    assert actual_weighted_f1 == approx(expected_stats["weighted_f1"])
    actual_a_f1 = get_stat_value(actual_stats, "classification_a_f1")
    assert actual_a_f1 == approx(expected_stats["a_f1"])
    actual_b_f1 = get_stat_value(actual_stats, "classification_b_f1")
    assert actual_b_f1 == approx(expected_stats["b_f1"])
    actual_c_f1 = get_stat_value(actual_stats, "classification_c_f1")
    assert actual_c_f1 == approx(expected_stats["c_f1"])
