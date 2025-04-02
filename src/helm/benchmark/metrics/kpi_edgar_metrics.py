from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple
import re

import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.evaluate_instances_metric import EvaluateInstancesMetric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.statistic import Stat
from helm.benchmark.scenarios.kpi_edgar_scenario import TAG_DICT


@dataclass
class _Entity:
    phrase: str
    tag: str


def _parse_entities(text: str) -> List[_Entity]:
    all_matches = re.findall(r"(?:^|(?<=\],))([^\[\]]+)\[([0-9a-z]+)\](?:$|(?=,))", text.strip())
    return [_Entity(phrase=match[0].strip(), tag=match[1].strip()) for match in all_matches]


def _compute_word_to_tags(entities: List[_Entity]) -> Dict[str, Set[str]]:
    word_to_tags: Dict[str, Set[str]] = defaultdict(set)
    for entity in entities:
        for word in entity.phrase.split():
            word_to_tags[word].add(entity.tag)
    return word_to_tags


def _compute_indicator_matrices(
    pred_entities: List[_Entity], gold_entities: List[_Entity], all_classes: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    pred_word_to_tags = _compute_word_to_tags(pred_entities)
    gold_word_to_tags = _compute_word_to_tags(gold_entities)

    all_words = list(pred_word_to_tags.keys() | gold_word_to_tags.keys())
    mlb = MultiLabelBinarizer().fit([all_classes])
    pred_matrix = mlb.transform([pred_word_to_tags[word] for word in all_words])
    gold_matrix = mlb.transform([gold_word_to_tags[word] for word in all_words])
    assert isinstance(pred_matrix, np.ndarray)
    assert isinstance(gold_matrix, np.ndarray)
    return (pred_matrix, gold_matrix)


def _get_request_state_indicator_matrices(
    request_state: RequestState, all_classes: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    assert request_state.result
    assert len(request_state.result.completions) == 1
    assert len(request_state.instance.references) == 1

    pred_text = request_state.instance.references[0].output.text
    gold_text = request_state.result.completions[0].text
    pred_entities = _parse_entities(pred_text)
    gold_entities = _parse_entities(gold_text)

    return _compute_indicator_matrices(
        pred_entities=pred_entities, gold_entities=gold_entities, all_classes=all_classes
    )


class EntityTypeClassificationMetric(EvaluateInstancesMetric):
    """Word-level entity type classification F1 score, macro-averaged across entity types."""

    def evaluate_instances(self, request_states: List[RequestState], eval_cache_path: str) -> List[Stat]:
        all_classes = list(TAG_DICT.keys())
        pred_indicator_matrixes: List[np.ndarray] = []
        gold_indicator_matrixes: List[np.ndarray] = []
        for request_state in request_states:
            pred_request_state_indicator_matrix, gold_request_state_indicator_matrix = (
                _get_request_state_indicator_matrices(request_state, all_classes)
            )
            pred_indicator_matrixes.append(pred_request_state_indicator_matrix)
            gold_indicator_matrixes.append(gold_request_state_indicator_matrix)
        pred_indicator_matrix = np.concatenate(pred_indicator_matrixes)
        gold_indicator_matrix = np.concatenate(gold_indicator_matrixes)
        computed_f1_score = f1_score(
            y_pred=pred_indicator_matrix, y_true=gold_indicator_matrix, average="macro", zero_division=0.0
        )
        return [Stat(MetricName("entity_type_classification_macro_f1")).add(computed_f1_score)]
