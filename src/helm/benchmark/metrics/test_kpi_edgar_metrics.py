from typing import List

from pytest import approx

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.kpi_edgar_metrics import KPIEdgarMetric, _request_states_to_pred_gold_pairs
from helm.benchmark.metrics.statistic import Stat
from helm.benchmark.scenarios.scenario import Input, Instance, Output, Reference, CORRECT_TAG
from helm.common.request import Request, RequestResult, GeneratedOutput


# The prediction under-predicts "kpi" (1 of 2 gold words) and misses "py" entirely, so the
# adjusted F1 (which normalizes by num_gold and num_pred separately) is not symmetric in
# (prediction, reference): the intended orientation gives 5/12, the swapped one gives 0.45.
_GOLD = "revenue increased[kpi], 100[cy], 90[py]"
_PRED = "revenue[kpi], 100[cy]"


def _request_state(prediction: str, reference: str) -> RequestState:
    return RequestState(
        instance=Instance(
            input=Input(text=""), references=[Reference(output=Output(text=reference), tags=[CORRECT_TAG])]
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


def _get_stat_value(stats: List[Stat], stat_name: str) -> float:
    for stat in stats:
        if stat.name.name == stat_name:
            assert stat.mean is not None
            return stat.mean
    raise ValueError(f"No stat with name {stat_name}")


def test_request_states_to_pred_gold_pairs_orders_prediction_before_reference():
    pairs = _request_states_to_pred_gold_pairs([_request_state(prediction=_PRED, reference=_GOLD)])
    assert pairs == [(_PRED, _GOLD)]


def test_kpi_edgar_metric_scores_prediction_against_reference():
    stats = KPIEdgarMetric().evaluate_instances([_request_state(prediction=_PRED, reference=_GOLD)], "")
    # Per tag (pred vs gold): kpi -> 2/3, cy -> 1, py -> 0, py1 -> 0; macro-averaged over the 4 tags.
    assert _get_stat_value(stats, "word_macro_f1_score") == approx(5 / 12)
    assert _get_stat_value(stats, "adjusted_macro_f1_score") == approx(5 / 12)
