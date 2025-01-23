from typing import List, Optional

from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.evaluate_instances_metric import EvaluateInstancesMetric
from helm.benchmark.metrics.evaluate_reference_metrics import normalize_text
from helm.benchmark.metrics.metric import MetricName
from helm.benchmark.metrics.statistic import Stat
from helm.benchmark.scenarios.scenario import Reference
from helm.common.hierarchical_logger import hlog
from helm.common.request import GeneratedOutput


def _normalize_label_text(text: str) -> str:
    return normalize_text(text, should_remove_articles=False)


class ClassificationMetric(EvaluateInstancesMetric):
    """Defines metrics for multi-class classification using the generation adapter.

    Currently provides `classification_macro_f1` and `classification_micro_f1`.
    These are population-level F1 measures to measure classification performance where each
    generation is a predicted class, and are different from the instance-level F1 measures
    in `BasicMetrics` that are intended to measure word overlap between the correct references
    and generations. The correct class should be provided by the normalized text of a correct
    reference. The predicted class for each instance is the normalized text of the generation.

    Note:
    - It is highly recommended to specify the set of classes should be specified using the
      `labels` parameter. Otherwise, the set of classes is derived from the correct references
      from all the instances. This means that classes may be incorrectly omitted if they are never
      used as a correct reference.
    - The `averages` parameter is a list of averaging methods to be used.
      It has the same meaning `average` as in scikit-learn.
    - Generations that are not in any of the known classes are counted as a
      negative prediction for every class.
    - Perturbed classes are considered different classes from unperturbed
      classes.
    - Currently, multi-label classification is not supported.
    """

    AVERAGE_OPTIONS = ["micro", "macro", "weighted", None]
    SCORE_OPTIONS = ["f1", "precision", "recall"]

    def __init__(
        self,
        averages: Optional[List[Optional[str]]] = None,
        labels: Optional[List[str]] = None,
        scores: Optional[List[str]] = None,
        delimiter: Optional[str] = None,
    ) -> None:
        """Creates metrics for multi-class classification.

        :param delimiter: For multi-label classification, the string delimiter between classes in the model's output.
        :param average: The list of scores to compute (e.g. "f1", "precision", "recall").
          Defaults to ["f1"].
        :param average: The averaging methods (e.g. "micro", "macro", "weighted") to be used.
          It has the same meaning `average` as in scikit-learn.
          Defaults to ["macro", "micro"].
        :param labels: The set of labels.
        :return: A list of `Stat` objects.
        """
        self.averages = averages or ["macro", "micro"]
        for average in self.averages:
            if average not in ClassificationMetric.AVERAGE_OPTIONS:
                raise ValueError(
                    f"Each value in `averages` must be set to one of {ClassificationMetric.AVERAGE_OPTIONS}."
                )
        self.scores = scores or ["f1"]
        for score_name in self.scores:
            if score_name not in ClassificationMetric.SCORE_OPTIONS:
                raise ValueError(f"Each value in `scores` must be set to one of {ClassificationMetric.SCORE_OPTIONS}.")
        self.delimiter = delimiter
        self.labels = labels
        if not self.labels:
            hlog(
                "WARNING: `labels` were not set on `ClassificationMetric`, "
                "so they will be inferred from target references. "
                "It is recommend to explicitly set `labels` on `ClassificationMetric`."
            )

    def is_multi_label(self) -> bool:
        return bool(self.delimiter)

    def evaluate_instances(self, request_states: List[RequestState], eval_cache_path: str) -> List[Stat]:
        y_pred: List[List[str]] = []
        y_true: List[List[str]] = []
        for request_state in request_states:  # one request state per instance
            # Only the generation adapter is supported.
            # For multiple_choice_* adapters, please use MultipleChoiceClassificationMetric.
            if request_state.reference_index is not None:
                raise ValueError("ClassificationMetric does not support multiple choice separate adapters")
            if request_state.request_mode == "calibration":
                raise ValueError("ClassificationMetric does not support calibration requests")
            assert request_state.result is not None
            if len(request_state.result.completions) != 1:
                raise ValueError("Result must contain exactly one completion")
            if request_state.output_mapping:
                raise ValueError("ClassificationMetric does not support multiple choice adapters")

            references = request_state.instance.all_correct_references
            if not self.is_multi_label():
                assert len(references) == 1
            correct_ref_texts = [_normalize_label_text(ref.output.text) for ref in references if ref.output.text]
            y_true.append(correct_ref_texts)

            input_text = request_state.result.completions[0].text
            predictions = input_text.split(self.delimiter) if self.is_multi_label() else [input_text]
            y_pred.append([_normalize_label_text(pred) for pred in predictions if pred])
        mlb = MultiLabelBinarizer().fit(
            [[_normalize_label_text(label) for label in self.labels]] if self.labels else y_true
        )
        y_true = mlb.transform(y_true)
        y_pred = mlb.transform(y_pred)
        stats: List[Stat] = []
        for average in self.averages:
            for score_name in self.scores:
                if score_name == "f1":
                    score_value = f1_score(y_pred=y_pred, y_true=y_true, average=average)
                elif score_name == "precision":
                    score_value = precision_score(y_pred=y_pred, y_true=y_true, average=average)
                elif score_name == "recall":
                    score_value = recall_score(y_pred=y_pred, y_true=y_true, average=average)
                else:
                    raise ValueError(
                        f"Unknown score name: '{score_name}' - expected one of ['f1', 'precision', 'recall']"
                    )
                if average is None:
                    for mlb_class, class_score_value in zip(mlb.classes_, score_value):
                        stats.append(
                            Stat(MetricName(f"classification_{mlb_class}_{score_name}")).add(class_score_value)
                        )
                else:
                    stats.append(Stat(MetricName(f"classification_{average}_{score_name}")).add(score_value))
        return stats


class MultipleChoiceClassificationMetric(EvaluateInstancesMetric):
    """
    Calculate population micro/macro F1 score for multiple_choice_* adapters.
    For generation adapters, please use ClassificationMetric.
    """

    def evaluate_instances(self, request_states: List[RequestState], eval_cache_path: str) -> List[Stat]:
        y_pred: List[str] = []
        y_true: List[str] = []
        for request_state in request_states:  # one request state per instance
            if request_state.request_mode == "calibration":
                raise ValueError("MultipleChoiceClassificationMetric does not support calibration requests")
            golds: List[Reference] = [
                reference for reference in request_state.instance.references if reference.is_correct
            ]
            assert len(golds) > 0, "MultipleChoiceClassificationMetric are designed for multiple_choice_* adapters"
            assert request_state.result is not None
            sorted_completions: List[GeneratedOutput] = sorted(
                request_state.result.completions, key=lambda x: -x.logprob
            )
            pred: str = sorted_completions[0].text.strip()  # Only utilize the first prediction
            if request_state.output_mapping is not None:
                pred = request_state.output_mapping.get(pred, pred)

            y_true.append(golds[0].output.text)
            y_pred.append(pred)

        return [
            Stat(MetricName("classification_macro_f1")).add(f1_score(y_pred=y_pred, y_true=y_true, average="macro")),
            Stat(MetricName("classification_micro_f1")).add(f1_score(y_pred=y_pred, y_true=y_true, average="micro")),
        ]
