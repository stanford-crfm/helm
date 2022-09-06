"""Evaluating source code generation."""

from typing import List, Union, Sequence, cast
import resource


from common.request import RequestResult
from benchmark.adapter import AdapterSpec, RequestState
from benchmark.scenarios.code_scenario import CodeReference
from . import code_metrics_helper
from .metric import Metric
from .metric_service import MetricService
from .metric_name import MetricName
from .statistic import Stat

MAXIMUM_MEMORY_BYTES = 8 * 1024 * 1024 * 1024  # 8GB.


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
    def __init__(self, names, timeout):
        super(APPSMetric, self).__init__()
        for name in names:
            if name not in METRICS.keys():
                raise ValueError(f"Expected name to be either one of {METRICS.keys()}, but got {name}.")
        self.names = names
        self.timeout = timeout

        # Set a memory limit for this process.
        resource.setrlimit(resource.RLIMIT_AS, (MAXIMUM_MEMORY_BYTES, MAXIMUM_MEMORY_BYTES))

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        reference = request_state.instance.references[0]
        reference = cast(CodeReference, reference)
        request_result: RequestResult = request_state.result
        metadata = reference.test_cases
        assert metadata is not None
        root = metadata.get("root")

        metrics = []
        for name in self.names:
            metric_fn = METRICS[name]

            best_score = 0.0
            for completion in request_result.completions:
                completion = completion.text.strip()
                scores = code_metrics_helper.run_test(root=root, test=completion, timeout=self.timeout)  # type: ignore
                scores = _convert_scores(scores)  # Convert list of bool/int to list of ints.
                this_score = metric_fn(scores)
                if this_score > best_score:
                    best_score = this_score
            metrics.append(Stat(MetricName(name)).add(best_score))
        return metrics
