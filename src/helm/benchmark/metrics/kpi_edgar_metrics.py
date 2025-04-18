from dataclasses import dataclass
from typing import Iterable, List, Dict, Set, Tuple
import re

import numpy as np

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.evaluate_instances_metric import EvaluateInstancesMetric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.statistic import Stat
from helm.benchmark.scenarios.kpi_edgar_scenario import KPIEDGARScenario


@dataclass
class _Entity:
    phrase: str
    tag: str


@dataclass
class _Counts:
    num_overlap: int
    num_pred: int
    num_gold: int


@dataclass
class _Rates:
    tp: float
    fn: float
    fp: float


def _get_all_tags() -> Iterable[str]:
    return KPIEDGARScenario.TAG_DICT.keys()


def _parse_entities(text: str) -> List[_Entity]:
    all_matches = re.findall(r"(?:^|(?<=\],))([^\[\]]+)\[([0-9a-z]+)\](?:$|(?=,))", text.strip())
    return [_Entity(phrase=match[0].strip(), tag=match[1].strip()) for match in all_matches]


def _compute_tag_to_words(entities: List[_Entity]) -> Dict[str, Set[str]]:
    tag_to_words: Dict[str, Set[str]] = {tag: set() for tag in _get_all_tags()}
    for entity in entities:
        for word in entity.phrase.split():
            tag_to_words[entity.tag].add(word)
    return tag_to_words


def _compute_tag_to_counts(pred_entities: List[_Entity], gold_entities: List[_Entity]) -> Dict[str, _Counts]:
    tag_to_counts: Dict[str, _Counts] = {}
    pred_tag_to_words = _compute_tag_to_words(pred_entities)
    gold_tag_to_words = _compute_tag_to_words(gold_entities)
    for tag in _get_all_tags():
        tag_to_counts[tag] = _Counts(
            num_overlap=len(pred_tag_to_words[tag] & gold_tag_to_words[tag]),
            num_pred=len(pred_tag_to_words[tag]),
            num_gold=len(gold_tag_to_words[tag]),
        )
    return tag_to_counts


def _counts_to_rates(counts: _Counts, adjust: bool) -> _Rates:
    if adjust:
        return _Rates(
            tp=counts.num_overlap / counts.num_gold if counts.num_gold > 0 else 0.0,
            fn=1 - (counts.num_overlap / counts.num_gold) if counts.num_gold > 0 else 0.0,
            fp=(counts.num_pred - counts.num_overlap) / counts.num_pred if counts.num_pred > 0 else 0.0,
        )
    else:
        return _Rates(
            tp=counts.num_overlap,
            fn=counts.num_gold - counts.num_overlap,
            fp=counts.num_pred - counts.num_overlap,
        )


def _compute_f1_score(rates: _Rates) -> float:
    return (2 * rates.tp) / (2 * rates.tp + rates.fp + rates.fn) if rates.tp + rates.fp + rates.fn > 0 else 0.0


def _compute_stats(pred_gold_pairs: List[Tuple[str, str]]) -> List[Stat]:
    tag_to_counts: Dict[str, _Counts] = {tag: _Counts(0, 0, 0) for tag in _get_all_tags()}
    for pred_text, gold_text in pred_gold_pairs:
        pred_entities = _parse_entities(pred_text)
        gold_entities = _parse_entities(gold_text)
        instance_tag_to_counts = _compute_tag_to_counts(pred_entities=pred_entities, gold_entities=gold_entities)
        for tag, instance_counts in instance_tag_to_counts.items():
            tag_to_counts[tag].num_overlap += instance_counts.num_overlap
            tag_to_counts[tag].num_pred += instance_counts.num_pred
            tag_to_counts[tag].num_gold += instance_counts.num_gold
    tag_word_f1_scores: List[float] = [
        _compute_f1_score(_counts_to_rates(counts, adjust=False)) for counts in tag_to_counts.values()
    ]
    tag_adjusted_f1_scores: List[float] = [
        _compute_f1_score(_counts_to_rates(counts, adjust=True)) for counts in tag_to_counts.values()
    ]
    return [
        Stat(MetricName("word_macro_f1_score")).add(np.mean(tag_word_f1_scores)),
        Stat(MetricName("adjusted_macro_f1_score")).add(np.mean(tag_adjusted_f1_scores)),
    ]


def _request_states_to_pred_gold_pairs(request_states: List[RequestState]) -> List[Tuple[str, str]]:
    pred_gold_pairs: List[Tuple[str, str]] = []
    for request_state in request_states:
        assert request_state.result
        assert len(request_state.result.completions) == 1
        assert len(request_state.instance.references) == 1
        pred_gold_pairs.append(
            (request_state.instance.references[0].output.text, request_state.result.completions[0].text)
        )
    return pred_gold_pairs


class KPIEdgarMetric(EvaluateInstancesMetric):
    """Word-level entity type classification F1 score, macro-averaged across entity types."""

    def evaluate_instances(self, request_states: List[RequestState], eval_cache_path: str) -> List[Stat]:
        return _compute_stats(_request_states_to_pred_gold_pairs(request_states))
