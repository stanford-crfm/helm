from typing import List, Dict, cast
from dataclasses import dataclass

from helm.common.general import parallel_map
from helm.common.request import Request
from helm.benchmark.adaptation.scenario_state import ScenarioState
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.statistic import Stat, merge_stat
from helm.benchmark.window_services.window_service import WindowService
from helm.benchmark.window_services.window_service_factory import WindowServiceFactory
from helm.benchmark.metrics.metric import MetricInterface, MetricResult, PerInstanceStats
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.tokens.auto_token_cost_estimator import AutoTokenCostEstimator
from helm.benchmark.metrics.tokens.token_cost_estimator import TokenCostEstimator


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
        # TODO: replace with "estimated_num_tokens" - is this for prompt or completion
        stats.append(Stat(MetricName("estimated_num_tokens_cost")).add(estimate_num_tokens_cost))

        stats.append(Stat(MetricName("num_completions")).add(request.num_completions))

        # Maximum number of tokens in the completions
        # This is an overestimate of the actual number of output tokens since sequences can early terminate
        stats.append(Stat(MetricName("max_num_completion_tokens")).add(request.num_completions * request.max_tokens))

        # Get number of tokens in the prompt
        tokenizer: WindowService = WindowServiceFactory.get_window_service(
            request.model_deployment, self.metric_service
        )
        num_prompt_tokens: int = tokenizer.get_num_tokens(request.prompt)
        stats.append(Stat(MetricName("num_prompt_tokens")).add(num_prompt_tokens))

        return stats


class DryRunMetric(MetricInterface):
    """Metrics for dry run."""

    def __init__(self):
        self.token_cost_estimator = AutoTokenCostEstimator()

    def __repr__(self):
        return "DryRunMetric"

    def evaluate(
        self,
        scenario_state: ScenarioState,
        metric_service: MetricService,
        eval_cache_path: str,
        parallelism: int,
    ) -> MetricResult:
        """
        Add up all the estimated number of tokens used for each request.
        """
        processor = Processor(token_cost_estimator=self.token_cost_estimator, metric_service=metric_service)
        results: List[List[Stat]] = parallel_map(
            processor.process,
            scenario_state.request_states,
            parallelism=parallelism,
        )

        # Per-instance
        per_instance_stats = [
            PerInstanceStats(
                cast(str, request_state.instance.id),
                request_state.instance.perturbation,
                request_state.train_trial_index,
                stats,
            )
            for request_state, stats in zip(scenario_state.request_states, results)
        ]

        # Aggregate
        stats: Dict[MetricName, Stat] = {}
        for instance_stats in results:
            for stat in instance_stats:
                merge_stat(stats, stat)

        merge_stat(stats, Stat(MetricName("num_requests")).add(len(scenario_state.request_states)))

        return MetricResult(list(stats.values()), per_instance_stats)
