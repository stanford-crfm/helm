from tqdm import tqdm
from typing import List, Dict, Tuple

from common.request import Request
from common.statistic import Stat, merge_stat
from proxy.tokenizer.auto_token_counter import AutoTokenCounter
from .adapter import ScenarioState
from .metric import Metric, MetricResult
from .metric_name import MetricName
from .metric_service import MetricService
from .scenario import Instance


class TokensMetric(Metric):
    """
    Estimates the total number of tokens that will be used based on the requests.
    """

    def __init__(self):
        self.token_counter = AutoTokenCounter()

    def evaluate(self, scenario_state: ScenarioState, metric_service: MetricService) -> MetricResult:
        """
        Add up all the estimated number of tokens used for each request.
        """
        per_instance_stats: Dict[Tuple[Instance, int], List[Stat]] = {}
        stats: Dict[MetricName, Stat] = {}

        for request_state in tqdm(scenario_state.request_states):
            request: Request = request_state.request
            num_tokens: int = self.token_counter.estimate_tokens(request)
            stat = Stat(MetricName("estimated_number_of_tokens")).add(num_tokens)
            merge_stat(stats, stat)
            per_instance_stats[(request_state.instance, 0)] = [stat]

        return MetricResult(list(stats.values()), per_instance_stats)
