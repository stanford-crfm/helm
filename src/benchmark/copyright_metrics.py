from typing import List

from nltk.tokenize.treebank import TreebankWordTokenizer

from common.statistic import Stat
from .adapter import AdapterSpec, RequestState
from .metric import Metric
from .metric_service import MetricService


def _longest_common_prefix_length(s1: List[str], s2: List[str]) -> int:
    min_len = min(len(s1), len(s2))
    for i in range(min_len):
        if s1[i] != s2[i]:
            return len(s1[:i])
    return min_len


class LongestCommonPrefixMetric(Metric):
    def __init__(self, normalize_by_prefix_length=False, **unused_kwargs):
        self.normalize_by_prefix_length = normalize_by_prefix_length
        self.tokenizer = TreebankWordTokenizer()
        del unused_kwargs

    def evaluate_generation(
        self, adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService
    ) -> List[Stat]:
        """Compute the length of the longest common prefix between reference and generations.

        Result is based on number of tokens produced with `nltk.tokenize.TreebankWordTokenizer`.
        When there are multiple generations, return the length of the longest.

        Example:
            input: A
            generations: [AABC, AMD]
            reference: AAD
            returns: 2
            explanation: The longest common prefix is AA (between AABC and AAD).
        """
        references = request_state.instance.references
        num_references = len(references)
        if num_references != 1:
            raise ValueError(
                f"Copyright scenario expects a single reference, but found {num_references} references."
            )
        prefix: str = request_state.instance.input
        reference: str = references[0].output[len(prefix):]

        result = 0
        for completion in request_state.result.completions:
            completion = completion.text.strip()
            truncated_reference = reference[:len(completion)]

            completion_tokens = self.tokenizer.tokenize(completion)
            truncated_reference_tokens = self.tokenizer.tokenize(truncated_reference)
            result = max(
                result, _longest_common_prefix_length(completion_tokens, truncated_reference_tokens)
            )

        if self.normalize_by_prefix_length:
            prefix_tokens = self.tokenizer.tokenize(prefix)
            result /= len(prefix_tokens)

        return [Stat('longest_common_prefix_length').add(result)]


class EditDistanceMetric(Metric):
    def __init__(self, **unused_kwargs):
        del unused_kwargs

    def evaluate_generation(
        self, adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService
    ) -> List[Stat]:
        """"""
        metrics = []
        return metrics
