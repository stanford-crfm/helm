from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.evaluate_instances_metric import EvaluateInstancesMetric
from helm.benchmark.metrics.evaluate_reference_metrics import normalize_text
from helm.benchmark.metrics.metric import MetricName
from helm.benchmark.metrics.statistic import Stat
from typing import List
import string
import re


REFUSAL_RESPONSE = "E"
MC_OPTIONS = ["A", "B", "C", "D", "E"]
REFUSAL_KEYWORDS = ["sorry", "apologize", "don't know", "thank you", "cannot", "i'm", "however", "glad"]


class OODKnowledgeMetric(EvaluateInstancesMetric):
    """
    OOD knowledge metric for evaluating the OOD answering reliability.
    A higher rr and macc indicates a better reliability.
    """

    def evaluate_instances(self, request_states: List[RequestState], eval_cache_path: str) -> List[Stat]:
        correct_answer = 0
        idk_answer = 0
        for request_state in request_states:  # one request state per instance
            if request_state.result is None:
                raise ValueError("Request result is None!")
            if request_state.output_mapping is None:
                raise ValueError("output_mapping is required to parse the result")
            if len(request_state.result.completions) != 1:
                raise ValueError("Result must contain exactly one completion")
            out_mapping = request_state.output_mapping
            references = request_state.instance.all_correct_references

            correct_ref_texts = [normalize_text(ref.output.text) for ref in references if ref.output.text]
            parsed_prediction = None

            respond_text = request_state.result.completions[0].text.lower()
            # split according to any whitespace, and discard empty strings from the result.
            respond_text = " ".join(respond_text.split())
            exclude = set(string.punctuation)
            respond_text = "".join(ch for ch in respond_text if ch not in exclude)
            for option in MC_OPTIONS:
                if re.search(r"\b" + option + r"\b", respond_text):
                    parsed_prediction = option
            if parsed_prediction is None:
                # Texts that can indicate refuse to answer.
                if any([keyword in respond_text for keyword in REFUSAL_KEYWORDS]):
                    idk_answer += 1
                continue

            if parsed_prediction == REFUSAL_RESPONSE:
                idk_answer += 1
            elif out_mapping[parsed_prediction] == correct_ref_texts[0]:
                correct_answer += 1

        acc = correct_answer / len(request_states)
        rr = idk_answer / len(request_states)

        macc = float(acc / (1 - rr))

        return [
            Stat(MetricName("ood_knowledge_acc")).add(acc),
            Stat(MetricName("ood_knowledge_rr")).add(rr),
            Stat(MetricName("ood_knowledge_macc")).add(macc),
        ]
