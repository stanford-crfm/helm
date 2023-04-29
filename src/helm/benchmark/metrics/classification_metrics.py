from typing import List, Optional

from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer

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
      This means that classes may be omitted if they are never used as a correct reference.
    - Generations that are not in any of the known classes are counted as a
      negative prediction for every class.
    - Perturbed classes are considered different classes from unperturbed
      classes.
    - Currently, multi-label classification is not supported.
    """

    def __init__(self, delimiter: Optional[str] = None):
        self.delimiter = delimiter

    def is_multi_label(self) -> bool:
        return bool(self.delimiter)

    def evaluate_instances(self, request_states: List[RequestState]) -> List[Stat]:
        y_pred: List[List[str]] = []
        y_true: List[List[str]] = []
        for request_state in request_states:  # one request state per instance
            # Only the generation adapter is supported.
            # TODO: Support multiple_choice_* adapters.
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
            correct_ref_texts = [normalize_text(ref.output.text) for ref in references if ref.output.text]
            y_true.append(correct_ref_texts)

            input_text = request_state.result.completions[0].text
            predictions = input_text.split(self.delimiter) if self.is_multi_label() else [input_text]
            y_pred.append([normalize_text(pred) for pred in predictions if pred])
        labels: List[str] = list(set(y for ys in y_true for y in ys))
        mlb = MultiLabelBinarizer().fit([labels])
        y_true = mlb.transform(y_true)
        y_pred = mlb.transform(y_pred)
        return [
            Stat(MetricName("classification_macro_f1")).add(f1_score(y_pred=y_pred, y_true=y_true, average="macro")),
            Stat(MetricName("classification_micro_f1")).add(f1_score(y_pred=y_pred, y_true=y_true, average="micro")),
        ]
