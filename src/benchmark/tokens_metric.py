from typing import List, Dict

from common.request import Request
from .adapter import ScenarioState
from .metrics.tokens.auto_token_cost_estimator import AutoTokenCostEstimator
from .metric import Metric, MetricResult, PerInstanceStatsKey
from .metric_name import MetricName
from .metric_service import MetricService
from .scenario import Instance
from .statistic import Stat, merge_stat
from .window_service.window_service import WindowService
from .window_service.window_service_factory import WindowServiceFactory


class TokensMetric(Metric):
    """
    Estimates the total number of tokens that will be used based on the requests.
    """

    def __init__(self):
        self.token_cost_estimator = AutoTokenCostEstimator()

    def evaluate(
        self, scenario_state: ScenarioState, metric_service: MetricService, eval_cache_path: str
    ) -> MetricResult:
        """
        Add up all the estimated number of tokens used for each request.
        """

        def merge_stat_helper(stat: Stat, instance: Instance):
            """Merges "stat" to `stats` and `per_instance_stats` dictionaries."""
            merge_stat(stats, stat)
            # Call take_mean to make a copy of the stat, so that merge_stat updates do
            # not change what is in per_instance_stats.
            per_instance_stats[PerInstanceStatsKey(instance, 0)] = [stat.take_mean()]

        per_instance_stats: Dict[PerInstanceStatsKey, List[Stat]] = {}
        stats: Dict[MetricName, Stat] = {}

        for request_state in scenario_state.request_states:
            request: Request = request_state.request
            instance: Instance = request_state.instance

            # Estimated cost in terms of number of tokens
            estimate_num_tokens_cost: int = self.token_cost_estimator.estimate_tokens(request, metric_service)
            stat = Stat(MetricName("estimated_num_tokens_cost")).add(estimate_num_tokens_cost)
            merge_stat_helper(stat, instance)

            # Number of tokens in the prompt
            window_service: WindowService = WindowServiceFactory.get_window_service(request.model, metric_service)
            num_prompt_tokens: int = window_service.get_num_tokens(text=request.prompt)
            stat = Stat(MetricName("num_prompt_tokens")).add(num_prompt_tokens)
            merge_stat_helper(stat, instance)

            # Maximum number of tokens in the completions
            stat = Stat(MetricName("max_num_output_tokens")).add(request.num_completions * request.max_tokens)
            merge_stat_helper(stat, instance)

        merge_stat(stats, Stat(MetricName("num_requests")).add(len(scenario_state.request_states)))
        return MetricResult(list(stats.values()), per_instance_stats)
