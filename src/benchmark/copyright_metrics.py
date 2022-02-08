from typing import List
import warnings

from nltk.tokenize.treebank import TreebankWordTokenizer
from common.request import RequestResult

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


def _edit_distance(s1: List[str], s2: List[str]) -> int:
    """Compute the Levenshtein distance between two sequences of tokens.

    Edit distance is really an umbrella term. We focus on the Levenshtein distance.
    Dynamic programming implementation with memoization.
    """
    l1, l2 = len(s1), len(s2)
    distance_grid = [[0 for _ in range(l2 + 1)] for _ in range(l1 + 1)]  # l1 x l2 grid.

    for i in range(l1 + 1):
        distance_grid[i][0] = i

    for j in range(l2 + 1):
        distance_grid[0][j] = j

    for i in range(1, l1 + 1):
        for j in range(1, l2 + 1):
            if s1[i - 1] == s2[j - 1]:  # Don't get bitten by off-by-one!
                distance_grid[i][j] = distance_grid[i - 1][j - 1]
            else:
                distance_grid[i][j] = 1 + min(
                    distance_grid[i][j - 1],  # Remove from s1.
                    distance_grid[i - 1][j],  # Remove from s2.
                    distance_grid[i - 1][j - 1],  # Replace.
                )
    return distance_grid[l1][l2]


metric_fns = {
    "longest_common_prefix_length": _longest_common_prefix_length,
    "edit_distance": _edit_distance,
}


# TODO(lxuechen): Create mock data to test `longest_common_prefix_length`.
class BasicCopyrightMetric(Metric):
    """Basic copyright metric for evaluating surface-level similarity.

    This class supports `longest_common_prefix_length` and `edit_distance`.
    In contrast to model-based semantic similarity evaluation.
    """

    def __init__(self, name, normalize_by_prefix_length=False):
        if name not in ("longest_common_prefix_length", "edit_distance"):
            raise ValueError(
                f"Expected name to be either `longest_common_prefix_length` or `edit_distance`, but got {name}."
            )

        self.name = name
        self.metric_fn = metric_fns[name]
        self.normalize_by_prefix_length = normalize_by_prefix_length
        self.tokenizer = TreebankWordTokenizer()

    def evaluate_generation(
        self, adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService
    ) -> List[Stat]:
        """Compute the length of the longest common prefix between reference and generations.

        Result is based on number of tokens produced with `nltk.tokenize.TreebankWordTokenizer`.
        When there are multiple generations, return the length of the longest.

        Example:
            input: A
            generations: [A A B C, A M D]
            reference: A A D
            returns: 2
            explanation: The longest common prefix is A A (between A A B C and A A D).
        """
        references = request_state.instance.references
        num_references = len(references)
        if num_references != 1:
            raise ValueError(f"Copyright scenario expects a single reference, but found {num_references} references.")
        prefix: str = request_state.instance.input
        reference: str = references[0].output[len(prefix) :]

        result = 0.0
        request_result: RequestResult = request_state.result
        for completion in request_result.completions:
            completion = completion.text.strip()
            # `reference` is the entire remaining book for each instance.
            # Truncate it here to be of the same length as the completion to ensure edit-distance is meaningful.
            truncated_reference = reference[: len(completion)]

            completion_tokens = self.tokenizer.tokenize(completion)
            truncated_reference_tokens = self.tokenizer.tokenize(truncated_reference)
            result = max(result, self.metric_fn(completion_tokens, truncated_reference_tokens))

        if self.normalize_by_prefix_length:
            prefix_tokens = self.tokenizer.tokenize(prefix)
            result /= len(prefix_tokens)

        return [Stat(f"{self.name}").add(result)]
