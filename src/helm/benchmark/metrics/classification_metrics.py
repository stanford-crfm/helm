from typing import List

from sklearn.metrics import f1_score

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.basic_metrics import normalize_text
from helm.benchmark.metrics.metric import Metric, MetricName
from helm.benchmark.metrics.statistic import Stat


class ClassificationMetric(Metric):
    """Defines metrics for multi-class classification using the generation adapter.

    Currently provides `classification_macro_f1` and `classification_micro_f1`.
    These are population-level F1 measures to measure classification performance where each
    generation is a predicted class, and are different from the instance-level F1 measures
    in `BasicMetrics` that are intended to measure word overlap between the correct references
    and generations. The correct class should be provided by the normalized text of a correct
    reference. The predicted class for each instance is the normalized text of the generation.

    Note:
    - The set of classes is derived from the correct references from all the instances.
      This means that classes may be omitted if they never are never used as a correct
      reference.
    - Generations that are not in any of the known classes are counted as a
      negative prediction for every class.
    - Perturbed classes are considered different classes from unperturbed
      classes.
    - Currently, multi-label classification is not supported.
    """

    def evaluate_instances(self, request_states: List[RequestState]) -> List[Stat]:
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
                if reference.is_correct:
                    num_correct += 1
                    y_true.append(normalize_text(reference.output.text))
            if num_correct != 1:
                # TODO: Support multi-label classification.
                raise ValueError("ClassificationMetric does not support multi-label classification")
            if request_state.output_mapping:
                raise ValueError("ClassificationMetric does not support multiple choice adapters")
            y_pred.append(normalize_text(request_state.result.completions[0].text))
        labels = list(set(y_true))
        return [
            Stat(MetricName("classification_macro_f1")).add(
                f1_score(y_pred=y_pred, y_true=y_true, labels=list(labels), average="macro")
            ),
            Stat(MetricName("classification_micro_f1")).add(
                f1_score(y_pred=y_pred, y_true=y_true, labels=list(labels), average="micro")
            ),
        ]
