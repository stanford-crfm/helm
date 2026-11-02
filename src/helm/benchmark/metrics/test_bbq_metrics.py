from types import SimpleNamespace

from helm.benchmark.metrics.bbq_metrics import BBQMetric


def _state(ambiguous, answer, gold):
    # reference tags end with: gold letter, target letter "A", non-target letter "B", unknown letter "C"
    tags = ["negative", "ambiguous" if ambiguous else "non-ambiguous", gold, "A", "B", "C"]
    return SimpleNamespace(
        instance=SimpleNamespace(references=[SimpleNamespace(tags=tags)]),
        result=SimpleNamespace(completions=[SimpleNamespace(text=answer)]),
    )


def _ambiguous_bias(num_disambiguated):
    # Two of four ambiguous questions get the stereotyped answer "A" (bias 1.0), the other two get "C".
    states = [_state(True, answer, "C") for answer in ["A", "A", "C", "C"]]
    states += [_state(False, "A", "A") for _ in range(num_disambiguated)]
    stats = BBQMetric().evaluate_instances(states, "")
    return {stat.name.name: stat.mean for stat in stats}["bbq_metric_ambiguous_bias"]


def test_ambiguous_bias_is_scaled_by_ambiguous_accuracy_only():
    # (1 - ambiguous accuracy) * bias = (1 - 0.5) * 1.0, whatever the disambiguated answers are
    assert _ambiguous_bias(4) == 0.5
    assert _ambiguous_bias(40) == 0.5
