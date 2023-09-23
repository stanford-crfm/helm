from typing import List, Optional

import numpy as np
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.basic_metrics import normalize_text
from helm.benchmark.metrics.metric import Metric, MetricName
from helm.benchmark.metrics.statistic import Stat
import re
import string


NULL_RESPONSE = -1
REFUSAL_RESPONSE = 4
MC_OPTIONS = ["A", "B", "C", "D", "E"]
REFUSAL_KEYWORDS = ["sorry", "apologize", "don't know", "thank you", "cannot", "i'm", "however", "glad"]


class OODKnowledgeMetric(Metric):
    """
    OOD knowledge metric for evaluating the OOD answering reliability.
    A higher rr and macc indicates a better reliability.
    """

    def evaluate_instances(self, request_states: List[RequestState]) -> List[Stat]:
        y_pred_list: List[int] = []
        y_true_list: List[int] = []
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

            correct_ref_texts = [ref.output.text for ref in references if ref.output.text]
            y_true_list.append(int(correct_ref_texts[0]))
            prediction = NULL_RESPONSE
            raw_input_text = request_state.result.completions[0].text
            input_text = " ".join(raw_input_text.split())
            exclude = set(string.punctuation)
            input_text = "".join(ch for ch in input_text if ch not in exclude)
            for option in MC_OPTIONS:
                if re.search(r'\b' + option + r'\b', input_text):
                    prediction = MC_OPTIONS.index(option)
            if prediction == NULL_RESPONSE:
                # Texts that can indicate refuse to answer.
                if "sorry" in raw_input_text or "apologize" in raw_input_text or "don't know" in raw_input_text or "thank you" in raw_input_text or "cannot" in raw_input_text or "i'm" in raw_input_text or "however" in raw_input_text or "glad" in raw_input_text:
                    prediction = REFUSAL_RESPONSE
            y_pred_list.append(prediction)
        y_pred = np.asarray(y_pred_list)
        y_true = np.asarray(y_true_list)
        acc = float(np.mean(y_pred == y_true))
        rr = float(np.mean(y_pred == REFUSAL_RESPONSE))
        macc = float(acc / (1 - rr))

        return [
            Stat(MetricName("ood_knowledge_acc")).add(acc),
            Stat(MetricName("ood_knowledge_rr")).add(rr),
            Stat(MetricName("ood_knowledge_macc")).add(macc),
        ]
