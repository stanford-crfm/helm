from collections import defaultdict
from tqdm import tqdm
from typing import Dict
import math
import numpy as np

from helm.common.request import RequestResult
from helm.benchmark.scenarios.scenario import Instance
from helm.benchmark.adaptation.scenario_state import ScenarioState
from helm.benchmark.metrics.statistic import Stat
from helm.benchmark.metrics.metric import MetricInterface, MetricResult
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService


class DenoisedRuntimeMetric(MetricInterface):
    def __repr__(self):
        return "DenoisedRuntimeMetric()"

    def evaluate(
        self,
        scenario_state: ScenarioState,
        metric_service: MetricService,
        eval_cache_path: str,
        parallelism: int,
    ) -> MetricResult:

        instance_to_min_request_times: Dict[Instance, float] = defaultdict(lambda: math.inf)
        for request_state in tqdm(scenario_state.request_states):
            assert request_state.result is not None
            request_result: RequestResult = request_state.result

            assert request_result.request_time is not None
            request_time: float = request_result.request_time

            instance: Instance = request_state.instance
            instance_to_min_request_times[instance] = min(instance_to_min_request_times[instance], request_time)

        denoised_runtime: float = float(np.mean(list(instance_to_min_request_times.values())))
        return MetricResult(
            aggregated_stats=[Stat(MetricName("denoised_runtime")).add(denoised_runtime)], per_instance_stats=[]
        )
