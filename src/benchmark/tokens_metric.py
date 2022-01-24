from typing import List, Dict

from common.request import TokenEstimationRequestResult
from common.statistic import Stat, merge_stat
from .adapter import ScenarioState
from .metric import Metric
from .metric_service import MetricService


class TokensMetric(Metric):
    """
    Defines metrics for tokens.
    """

    def evaluate(self, scenario_state: ScenarioState, metric_service: MetricService) -> List[Stat]:
        """
        Main entry point for a `Metric`.  This function groups the the single
        list of `RequestState` by training trial and instance, and invokes
        other functions to process those.  This should serve most purposes.

        Any logic that doesn't decompose along instances should go here, such
        as robustness.
        """
        stats: Dict[str, Stat] = {}

        for request_state in scenario_state.request_states:
            result: TokenEstimationRequestResult = metric_service.estimate_tokens(request_state.request)
            merge_stat(stats, Stat(f"{result.model_group}_estimated_number_of_tokens").add(result.num_tokens))

        return list(stats.values())
