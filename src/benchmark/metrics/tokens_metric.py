from typing import List, Dict
from dataclasses import dataclass

from common.general import parallel_map
from common.request import Request
from benchmark.adapter import ScenarioState, RequestState
from benchmark.metrics.statistic import Stat, merge_stat
from benchmark.window_services.window_service import WindowService
from benchmark.window_services.window_service_factory import WindowServiceFactory
from .metric import Metric, MetricResult, PerInstanceStatsKey
from .metric_name import MetricName
from .metric_service import MetricService
from .tokens.auto_token_cost_estimator import AutoTokenCostEstimator
from .tokens.token_cost_estimator import TokenCostEstimator


@dataclass
class Processor:
    """Processes a single example."""

    token_cost_estimator: TokenCostEstimator
    metric_service: MetricService

    def process(self, request_state: RequestState) -> List[Stat]:
        request: Request = request_state.request
        stats: List[Stat] = []

        # Estimated cost in terms of number of tokens
        estimate_num_tokens_cost: int = self.token_cost_estimator.estimate_tokens(request, self.metric_service)
        stats.append(Stat(MetricName("estimated_num_tokens_cost")).add(estimate_num_tokens_cost))

        # Number of tokens in the prompt
        window_service: WindowService = WindowServiceFactory.get_window_service(request.model, self.metric_service)
        num_prompt_tokens: int = window_service.get_num_tokens(text=request.prompt)
        stats.append(Stat(MetricName("num_prompt_tokens")).add(num_prompt_tokens))

        # Number of characters in the prompt
        stats.append(Stat(MetricName("num_prompt_characters")).add(len(request.prompt)))

        # Maximum number of tokens in the completions
        # TODO: this is an overestimate since stop sequences can cause early termination.
        stats.append(Stat(MetricName("max_num_output_tokens")).add(request.num_completions * request.max_tokens))

        return stats


class TokensMetric(Metric):
    """
    Estimates the total number of tokens that will be used based on the requests.
    """

    def __init__(self):
        self.token_cost_estimator = AutoTokenCostEstimator()

    def __repr__(self):
        return "TokensMetric()"

    def evaluate(
        self, scenario_state: ScenarioState, metric_service: MetricService, eval_cache_path: str, parallelism: int,
    ) -> MetricResult:
        """
        Add up all the estimated number of tokens used for each request.
        """
        processor = Processor(token_cost_estimator=self.token_cost_estimator, metric_service=metric_service)
        results: List[List[Stat]] = parallel_map(
            processor.process, scenario_state.request_states, parallelism=parallelism,
        )

        # Per instance
        keys = [PerInstanceStatsKey(instance, 0) for instance in scenario_state.instances]
        per_instance_stats: Dict[PerInstanceStatsKey, List[Stat]] = dict(zip(keys, results))

        # Aggregate
        stats: Dict[MetricName, Stat] = {}
        for instance_stats in results:
            for stat in instance_stats:
                merge_stat(stats, stat)

        merge_stat(stats, Stat(MetricName("num_requests")).add(len(scenario_state.request_states)))

        return MetricResult(list(stats.values()), per_instance_stats)
