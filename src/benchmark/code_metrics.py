"""Evaluating source code generation."""

from typing import List, Union, Sequence

from common.request import RequestResult
from common.statistic import Stat
from . import code_metrics_helper
from .adapter import AdapterSpec, RequestState
from .metric import Metric
from .metric_service import MetricService


def _convert_scores(scores: Sequence[Union[int, bool]]):
    """Convert boolean scores to int."""
    # `scores` is returned by `run_test`, and is a list of bools/ints.
    return [1.0 if isinstance(score, bool) and score else 0.0 for score in scores]


def _test_avg(scores: Sequence[Union[int, bool]]) -> float:
    """Compute the average number of tests passed."""
    scores = _convert_scores(scores)
    # Division by zero should not be a convern, given our data processing.
    return sum(scores) / len(scores)


def _strict_acc(scores: Sequence[Union[int, bool]]) -> float:
    scores = _convert_scores(scores)
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
        request_result: RequestResult = request_state.result
        # Path to folder of the instance, with files like input_output.json.
        if isinstance(instance.data, dict):
            root = instance.data["root"]
        else:
            raise ValueError(f"Expected path in `instance.data`, but found instance to be {instance}.")

        metrics = []
        for name in self.names:
            metric_fn = METRICS[name]

            score = 0.0
            for completion in request_result.completions:
                completion = completion.text.strip()
                scores = code_metrics_helper.run_test(root=root, test=completion)
                metric = metric_fn(scores)
                if metric > score:
                    score = metric
            metrics.append(Stat(name).add(score))
        return metrics
