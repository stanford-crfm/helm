"""Evaluating source code generation."""

from typing import List, Union, Sequence, cast

from common.request import RequestResult
from common.statistic import Stat
from . import code_metrics_helper
from .adapter import AdapterSpec, RequestState
from .metric import Metric
from .metric_service import MetricService
from .code_scenario import CodeInstance


def _convert_scores(scores: Sequence[Union[int, bool]]) -> List[float]:
    """Convert boolean scores to int."""
    # `scores` is returned by `code_metrics_helper.run_test` and is a list of bools/ints.
    return [1.0 if isinstance(score, bool) and score else 0.0 for score in scores]


def _test_avg(scores: List[float]) -> float:
    """Compute the average number of tests passed."""
    # Division by zero should not be a concern, given our data processing.
    return sum(scores) / len(scores)


def _strict_acc(scores: List[float]) -> float:
    """Return 1.0 if all tests passed; otherwise return 0.0."""
    return 1.0 if sum(scores) == len(scores) else 0.0


METRICS = {
    "test_avg": _test_avg,
    "strict_acc": _strict_acc,
}


class APPSMetric(Metric):
    def __init__(self, names):
        super(APPSMetric, self).__init__()
        for name in names:
            if name not in METRICS.keys():
                raise ValueError(f"Expected name to be either one of {METRICS.keys()}, but got {name}.")
        self.names = names

    def evaluate_generation(
        self, adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService
    ) -> List[Stat]:
        instance = request_state.instance
        instance = cast(CodeInstance, instance)
        request_result: RequestResult = request_state.result
        # Type cast Optional[Dict] to Dict; we know for sure it's not None for this scenario.
        root = cast(dict, instance.metadata).get("root")

        metrics = []
        for name in self.names:
            metric_fn = METRICS[name]

            best_score = 0.0
            for completion in request_result.completions:
                completion = completion.text.strip()
                scores = code_metrics_helper.run_test(root=root, test=completion)  # type: ignore
                scores = _convert_scores(scores)  # Convert list of bool/int to list of ints.
                this_score = metric_fn(scores)
                if this_score > best_score:
                    best_score = this_score
            metrics.append(Stat(name).add(best_score))
        return metrics
