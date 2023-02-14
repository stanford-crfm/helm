from typing import List, Set

from sklearn.metrics import f1_score

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.basic_metrics import normalize_text
from helm.benchmark.metrics.metric import Metric, MetricName
from helm.benchmark.metrics.statistic import Stat


class ClassificationMetric(Metric):
    def evaluate_instances(self, request_states: List[RequestState]) -> List[Stat]:
        normalized_labels: Set[str] = set()
        y_pred: List[str] = []
        y_true: List[str] = []
        for request_state in request_states:
            # Only the generation adapter is supported.
            # TODO: Support multiple_choice_* adapters.
            if request_state.reference_index is not None:
                raise ValueError("ClassificationMetric does not support multiple choice separate adapters")
            if request_state.request_mode == "calibration":
                raise ValueError("ClassificationMetric does not support calibration requests")
            assert request_state.result is not None
            if len(request_state.result.completions) != 1:
                raise ValueError("Result must contain exactly one completion")

            num_correct = 0
            for reference in request_state.instance.references:
                normalized_labels.add(normalize_text(reference.output.text))
                if reference.is_correct:
                    num_correct += 1
                    y_true.append(normalize_text(reference.output.text))
            if num_correct != 1:
                # TODO: Support multi-label classification.
                raise ValueError("ClassificationMetric does not support multi-label classification")
            if request_state.output_mapping is not None:
                raise ValueError("ClassificationMetric does not support multiple choice adapters")
            y_pred.append(normalize_text(request_state.result.completions[0].text))
        # TODO: Support F1 on perturbation slices.
        return [
            Stat(MetricName("classification_macro_f1")).add(
                f1_score(y_pred=y_pred, y_true=y_true, labels=list(normalized_labels), average="macro")
            ),
            Stat(MetricName("classification_micro_f1")).add(
                f1_score(y_pred=y_pred, y_true=y_true, labels=list(normalized_labels), average="micro")
            ),
        ]
