from typing import List

import numpy as np

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.evaluate_instances_metric import EvaluateInstancesMetric
from helm.benchmark.metrics.metric import MetricName
from helm.benchmark.metrics.statistic import Stat
from helm.common.request import GeneratedOutput


class CLEVATopKAccuracyMetric(EvaluateInstancesMetric):
    """Defines metrics for CLEVA conceptual generalization task.

    This is not a conventional accuracy@k metric but rather a special one taken from
    https://openreview.net/pdf?id=gJcEM8sxHK

    It accepts multiple predictions and multiple references to calculate the accuracy
    per instance. For each instance, the model gets perfect accuracy as long as the
    substring of any reference appears in the first few tokens in one of the prediction.
    """

    def __init__(self, k: int, cut_off: int):
        self.k = k
        self.cut_off = cut_off

    def correct_or_not(self, completions: List[str], references: List[str]) -> bool:
        for prediction in completions[: self.k]:
            prediction_text: str = prediction[: self.cut_off]
            for reference_text in references:
                for start in range(len(reference_text)):
                    for end in range(start + 1, len(reference_text) + 1):
                        reference_substring = reference_text[start:end]
                        if reference_substring in prediction_text:
                            # we will consider the prediction correct as long as
                            # a substring of any possible reference appears in it
                            return True
        return False

    def evaluate_instances(self, request_states: List[RequestState], eval_cache_path: str) -> List[Stat]:
        per_instance_accuracy: List[bool] = []
        for request_state in request_states:  # one request state per instance
            assert request_state.result is not None
            references = request_state.instance.all_correct_references
            correct_ref_texts = [ref.output.text for ref in references if ref.output.text]

            sorted_completions: List[GeneratedOutput] = sorted(
                request_state.result.completions, key=lambda x: -x.logprob
            )
            sorted_completions_text: List[str] = [completion.text for completion in sorted_completions]
            correct = self.correct_or_not(sorted_completions_text, correct_ref_texts)
            per_instance_accuracy.append(correct)
        accuracy: float = np.mean(np.asarray(per_instance_accuracy, dtype=np.float32)).item()

        return [
            Stat(MetricName(f"cleva_top{self.k}_accuracy")).add(accuracy),
        ]
