"""Evaluating source code generation."""

import threading
import multiprocessing
from typing import List, Union, Sequence, cast

from helm.common.hierarchical_logger import hlog
from helm.common.request import RequestResult
from helm.benchmark.adaptation.scenario_state import ScenarioState
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.scenarios.code_scenario import CodeReference
from helm.benchmark.metrics import code_metrics_helper
from helm.benchmark.metrics.metric import Metric, MetricResult
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.statistic import Stat

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


def _run_test_wrapper(root: str, test: str, timeout: float, shared_list: list):
    shared_list.append(code_metrics_helper.run_test(root, test, timeout))  # type: ignore


class APPSMetric(Metric):
    def __init__(self, names, timeout):
        super(APPSMetric, self).__init__()
        for name in names:
            if name not in METRICS.keys():
                raise ValueError(f"Expected name to be either one of {METRICS.keys()}, but got {name}.")
        self.names = names
        self.timeout = timeout

        # Set a memory limit for this process.
        # TODO: debugging - remove this later -Chen
        # resource.setrlimit(resource.RLIMIT_AS, (MAXIMUM_MEMORY_BYTES, MAXIMUM_MEMORY_BYTES))

    def evaluate(
        self, scenario_state: ScenarioState, metric_service: MetricService, eval_cache_path: str, parallelism: int
    ) -> MetricResult:
        # Running with parallelism > 1 causes the run to get stuck.
        hlog(
            f"Setting parallelism from {parallelism} to 1, since evaluating code with parallelism > 1 isn't supported."
        )
        return super().evaluate(scenario_state, metric_service, eval_cache_path, parallelism=1)

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        reference = request_state.instance.references[0]
        reference = cast(CodeReference, reference)
        assert request_state.result is not None
        request_result: RequestResult = request_state.result
        metadata = reference.test_cases
        assert metadata is not None
        root = metadata.get("root")

        metrics = []
        for name in self.names:
            metric_fn = METRICS[name]

            best_score = 0.0
            for completion_sequence in request_result.completions:
                completion = completion_sequence.text.strip()

                # Similar to the logic in https://github.com/hendrycks/apps/blob/main/eval/test_one_solution.py
                # Running the testing code in a forked process prevents against annoying memory issues.
                shared_list = multiprocessing.Manager().list()  # Create shared object to hold results.
                p = multiprocessing.Process(
                    target=_run_test_wrapper, args=(root, completion, self.timeout, shared_list)
                )
                p.start()
                p.join(timeout=11)  # Same 'global' timeout used in original APPS codebase.
                if p.is_alive():
                    hlog(f"Before kill thread count: {threading.active_count()} exitcode: {p.exitcode}")
                    p.kill()
                    p.join(timeout=60)
                    hlog(f"After second join thread count: {threading.active_count()}. exitcode: {p.exitcode}")
                    assert not p.is_alive(), "The code process was still alive even after calling kill."

                if len(shared_list) > 0:
                    scores = shared_list[0]
                else:
                    # Remark: ideally should consider all tests that failed;
                    # use the average number of tests here for simplicity
                    avg_number_tests = 21
                    scores = [-1] * avg_number_tests

                scores = _convert_scores(scores)  # Convert list of bool/int to list of ints.
                this_score = metric_fn(scores)
                if this_score > best_score:
                    best_score = this_score
            metrics.append(Stat(MetricName(name)).add(best_score))
        return metrics
