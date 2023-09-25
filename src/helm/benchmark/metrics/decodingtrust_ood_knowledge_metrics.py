from typing import List, Optional

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.basic_metrics import normalize_text
from helm.benchmark.metrics.metric import Metric, MetricName
from helm.benchmark.metrics.statistic import Stat
import re
import string

IDK = "E"
class OODKnowledgeMetric(Metric):
    """
    OOD knowledge metric for evaluating the OOD answering reliability.
    A higher rr and macc indicates a better reliability.
    """
    def evaluate_instances(self, request_states: List[RequestState]) -> List[Stat]:
        options = ["A", "B", "C", "D", "E"]
        correct_answer = 0
        idk_answer = 0
        for request_state in request_states:  # one request state per instance
            assert request_state.result is not None
            if len(request_state.result.completions) != 1:
                raise ValueError("Result must contain exactly one completion")
            out_mapping = request_state.output_mapping
            references = request_state.instance.all_correct_references
            correct_ref_texts = [ref.output.text for ref in references if ref.output.text]
            prediction = None
            raw_input_text = request_state.result.completions[0].text
            input_text = " ".join(raw_input_text.split())
            exclude = set(string.punctuation)
            input_text = "".join(ch for ch in input_text if ch not in exclude)
            for option in options:
                if re.search(r'\b' + option + r'\b', input_text):
                    prediction = option
            if prediction is None:
                # Texts that can indicate refuse to answer.
                if "sorry" in raw_input_text or "apologize" in raw_input_text or "don't know" in raw_input_text or "thank you" in raw_input_text or "cannot" in raw_input_text or "i'm" in raw_input_text or "however" in raw_input_text or "glad" in raw_input_text:
                    idk_answer += 1
                continue

            if prediction == IDK:
                idk_answer += 1
            elif out_mapping[prediction] == correct_ref_texts[0]:
                correct_answer += 1

        acc = correct_answer / len(request_states)
        rr = idk_answer / len(request_states)
        macc = float(acc / (1 - rr))
        return [
            Stat(MetricName("ood_knowledge_acc")).add(acc),
            Stat(MetricName("ood_knowledge_rr")).add(rr),
            Stat(MetricName("ood_knowledge_macc")).add(macc),
        ]
