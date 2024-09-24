from typing import List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


class AnnotationLabelMetric(Metric):
    """Binary metric for labels produced by annotators.

    Expects the annotation with the given annotator name and key to be a string label.

    For each possible label in the list of possible labels, produces a
    corresponding stat with a value of 1 or 0 indicating if the actual label
    in the annoation."""

    def __init__(self, annotator_name: str, key: str, labels: List[str]):
        super().__init__()
        self.annotator_name = annotator_name
        self.key = key
        self.labels = labels

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.annotations
        annotation_label = request_state.annotations[self.annotator_name][self.key]
        if annotation_label not in self.labels:
            raise ValueError(
                f"Unrecognized annotation label '{annotation_label}' "
                f"(known labels: {self.labels}) "
                f"in annotation {request_state.annotations[self.annotator_name]} "
                f"for instance id {request_state.instance.id}"
            )
        stats: List[Stat] = []
        for label in self.labels:
            stats.append(
                Stat(MetricName(f"annotation_{self.annotator_name}_{self.key}_{label}")).add(
                    1 if label == annotation_label else 0
                )
            )
        return stats


class AnnotationNumericMetric(Metric):
    """Numeric metric for numbers produced by annotators.

    Expects the annotation with the given annotator name and key to be a number."""

    def __init__(self, annotator_name: str, key: str):
        super().__init__()
        self.annotator_name = annotator_name
        self.key = key

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.annotations
        score = request_state.annotations[self.annotator_name][self.key]
        return [Stat(MetricName(f"annotation_{self.annotator_name}_{self.key}")).add(score)]


class AnnotationLikertScaleMetric(Metric):
    """Numeric metric for labels produced by annotators.

    Expects the annotation with the given annotator name and key to be a string label.

    For each possible label in the list of possible labels, produces a
    corresponding stat with a value of 1 or 0 indicating if the actual label
    in the annoation."""

    def __init__(self, annotator_name: str, key: str, min_score: int, max_score: int):
        super().__init__()
        self.annotator_name = annotator_name
        self.key = key
        self.min_score = min_score
        self.max_score = max_score

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.annotations
        likert_score = request_state.annotations[self.annotator_name][self.key]
        if likert_score < self.min_score or likert_score > self.max_score:
            raise ValueError(
                f"Likert score {likert_score} "
                f"out of bounds {self.min_score} to {self.max_score} "
                f"under key {self.key} and annotator {self.annotator_name} "
                f"in annotation {request_state.annotations[self.annotator_name]} "
                f"for instance id {request_state.instance.id}"
            )
        normalized_score = (likert_score - self.min_score) / (self.max_score - self.min_score)
        return [Stat(MetricName(f"annotation_{self.annotator_name}_{self.key}")).add(normalized_score)]
