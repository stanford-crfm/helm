from typing import List, Optional

import numpy as np

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.basic_metrics import normalize_text
from helm.benchmark.metrics.metric import Metric, MetricName
from helm.benchmark.metrics.statistic import Stat


class OODKnowledgeMetric(Metric):
    """
    OOD knowledge metric for evaluating the OOD answering reliability.
    A higher rr and macc indicates a better reliability.
    """
    def evaluate_instances(self, request_states: List[RequestState]) -> List[Stat]:
        y_pred: List[int] = []
        y_true: List[int] = []
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
            assert len(references) == 1
            correct_ref_texts = [normalize_text(ref.output.text) for ref in references if ref.output.text]
            y_true.append(int(correct_ref_texts[0]))

            input_text = request_state.result.completions[0].text
            predictions = [input_text]
            for pred in predictions:
                digit = -1
                for pred_char in pred.split("202")[0]:
                    if pred_char.isdigit():
                        digit = int(pred_char)
                        break
                if digit == -1:
                    if "sorry" in pred or "apologize" in pred or "don't know" in pred or "thank you" in pred or "cannot" in pred or "i'm" in pred or "however" in pred or "glad" in pred:
                        digit = 4
                y_pred.append(digit)
        y_pred = np.asarray(y_pred)
        y_true = np.asarray(y_true)
        acc = float(np.mean(y_pred == y_true))
        rr = float(np.mean(y_pred == 4))
        macc = float(acc / (1 - rr))
        
        return [
            Stat(MetricName("ood_knowledge_acc")).add(acc),
            Stat(MetricName("ood_knowledge_rr")).add(rr),
            Stat(MetricName("ood_knowledge_macc")).add(macc),
        ]
