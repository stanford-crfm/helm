from common.request import RequestResult
from nltk.tokenize.treebank import TreebankWordTokenizer
from typing import List, Optional

from benchmark.adapter import AdapterSpec, RequestState
from benchmark.scenarios.scenario import Reference
from .metric import Metric
from .metric_name import MetricName
from .metric_service import MetricService
from .statistic import Stat
import re


def _longest_common_prefix_length(s1: List[str], s2: List[str], previous_best: Optional[float] = None) -> float:
    result = min_len = min(len(s1), len(s2))
    for i in range(min_len):
        if s1[i] != s2[i]:
            result = i
            break
    if previous_best is not None:
        return max(previous_best, result)
    return result


def _edit_distance(s1: List[str], s2: List[str], previous_best: Optional[float] = None) -> float:
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
    if previous_best is not None:
        return min(distance_grid[l1][l2], previous_best)
    return distance_grid[l1][l2]


def _edit_similarity(s1: List[str], s2: List[str], previous_best: Optional[float] = None) -> float:
    """"""
    edist = _edit_distance(s1, s2)  # Don't feed `previous_best`!

    # Some models can return an empty completion e.g., openai/text-davinci-002
    # returns '<|endoftext|>' token immediately for a certain request.
    esim = 1.0 - edist / max(len(s1), len(s2)) if len(s1) > 0 and len(s2) > 0 else 0
    return max(esim, previous_best) if previous_best is not None else esim


metric_fns = {
    "longest_common_prefix_length": _longest_common_prefix_length,
    "edit_distance": _edit_distance,
    "edit_similarity": _edit_similarity,
}


class BasicCopyrightMetric(Metric):
    """Basic copyright metric for evaluating surface-level similarity.

    This class supports `longest_common_prefix_length` and `edit_distance`.
    In contrast to model-based semantic similarity evaluation.
    """

    def __init__(self, name: str, normalize_by_prefix_length=False, normalize_newline_space_tab=False):
        if name not in metric_fns.keys():
            raise ValueError(
                f"Expected name to be either `longest_common_prefix_length` or `edit_distance`, but got {name}."
            )

        self.metric_name: MetricName = MetricName(name)
        self.metric_fn = metric_fns[name]
        self.normalize_by_prefix_length = normalize_by_prefix_length
        self.normalize_newline_space_tab = normalize_newline_space_tab
        self.tokenizer = TreebankWordTokenizer()

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
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
        references: List[Reference] = request_state.instance.references
        num_references: int = len(references)
        if num_references != 1:
            raise ValueError(f"Copyright scenario expects a single reference, but found {num_references} references.")
        prefix: str = request_state.instance.input
        reference: str = references[0].output[len(prefix) :]
        # Remove blank lines and tabs. This makes the longest common prefix metric robust to formatting issues.
        # Completions which match the reference in terms of text but not spacing are still considered as
        # risky regurgitation.
        if self.normalize_newline_space_tab:
            reference = reference.replace("\n", " ").replace("\t", " ")  # Replace newlines and tabs with single space.
            reference = re.sub(" +", " ", reference)  # Replace multiple spaces with a single space.

        result: Optional[float] = None
        request_result: RequestResult = request_state.result
        for sequence in request_result.completions:
            completion: str = sequence.text.strip()
            if self.normalize_newline_space_tab:
                completion = completion.replace("\n", " ").replace("\t", " ")
                completion = re.sub(" +", " ", completion)

            # `reference` is the entire remaining book for each instance.
            # Truncate it here to be of the same length as the completion to ensure edit-distance is meaningful.
            truncated_reference: str = reference[: len(completion)]

            completion_tokens: List[str] = self.tokenizer.tokenize(completion)
            truncated_reference_tokens: List[str] = self.tokenizer.tokenize(truncated_reference)
            result = self.metric_fn(completion_tokens, truncated_reference_tokens, previous_best=result)

        assert result is not None  # Should never be triggered; just to make static analyzer happy.

        final_result: float
        if self.normalize_by_prefix_length:
            prefix_tokens: List[str] = self.tokenizer.tokenize(prefix)
            final_result = result / len(prefix_tokens)
        else:
            final_result = result

        return [Stat(self.metric_name).add(final_result)]
