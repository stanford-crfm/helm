from typing import List

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.statistic import Stat
from helm.benchmark.metrics.evaluate_reference_metrics import normalize_text
from helm.benchmark.metrics.evaluate_instances_metric import EvaluateInstancesMetric
from helm.benchmark.scenarios.scenario import (
    CORRECT_TAG,
)
from sklearn.metrics import f1_score, accuracy_score


class UltraSuiteASRMetric(EvaluateInstancesMetric):
    """Score metrics for UltraSuite ASR."""

    def evaluate_instances(self, request_states: List[RequestState], eval_cache_path: str) -> List[Stat]:
        y_pred: List[str] = []
        y_pred_quasi: List[str] = []
        y_true: List[str] = []
        for request_state in request_states:  # one request state per instance

            for reference in request_state.instance.references:
                if reference.tags == [CORRECT_TAG]:
                    true_label = reference.output.text
                    break

            assert request_state.result
            model_output_text = request_state.result.completions[0].text.strip().lower()
            assert request_state.instance.extra_data
            ground_truth_text = request_state.instance.extra_data["transcription"].strip().lower()

            if model_output_text == ground_truth_text:
                predicted_label = "typically_developing"
            else:
                predicted_label = "speech_disorder"

            if normalize_text(predicted_label) == normalize_text(true_label):
                quasi_label = "typically_developing"
            else:
                quasi_label = "speech_disorder"

            y_true.append(true_label)
            y_pred.append(predicted_label)
            y_pred_quasi.append(quasi_label)

        return [
            Stat(MetricName("classification_macro_f1")).add(f1_score(y_pred=y_pred, y_true=y_true, average="macro")),
            Stat(MetricName("classification_micro_f1")).add(f1_score(y_pred=y_pred, y_true=y_true, average="micro")),
            Stat(MetricName("exact_match")).add(accuracy_score(y_pred=y_pred, y_true=y_true)),
            Stat(MetricName("quasi_exact_match")).add(accuracy_score(y_pred=y_pred_quasi, y_true=y_true)),
        ]
