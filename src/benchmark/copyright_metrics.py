from typing import List

from common.statistic import Stat
from .adapter import AdapterSpec, RequestState
from .metric import Metric
from .metric_service import MetricService


def _longest_common_prefix_length(s1: str, s2: str) -> int:
    min_len = min(len(s1), len(s2))
    for i in range(min_len):
        if s1[i] != s2[i]:
            return len(s1[:i])
    return min_len


class LongestCommonPrefixLengthMetric(Metric):
    def __init__(self, **unused_kwargs):
        del unused_kwargs

    def evaluate_generation(
        self, adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService
    ) -> List[Stat]:
        """Compute the length of the longest common prefix between reference and generation.

        When there are multiple references, return the length of the longest.
        E.g., inputs ref=AABC, gen=[AACD, ACDD] give the result of 2.
        """
        references = request_state.instance.references
        num_references = len(references)
        if num_references != 1:
            raise ValueError(
                f"Copyright scenario expects a single reference, but found {num_references} references."
            )
        reference: str = references[0].output

        # TODO: Develop a test for this.
        result = 0
        for completion in request_state.result.completions:
            completion = completion.text.strip()
            truncated_reference = reference[:len(completion)]
            result = max(result, _longest_common_prefix_length(completion, truncated_reference))

        return [Stat('longest_common_prefix_length').add(result)]


class EditDistanceMetric(Metric):
    def __init__(self, **unused_kwargs):
        del unused_kwargs

    def evaluate_generation(
        self, adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService
    ) -> List[Stat]:
        """Compute the reference metrics and language modeling metrics"""
        metrics = []
        return metrics
