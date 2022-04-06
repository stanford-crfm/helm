from typing import List

from common.request import RequestResult
from common.statistic import Stat
from .adapter import AdapterSpec, RequestState
from .metric import Metric
from .metric_name import MetricName
from .metric_service import MetricService

from .numeracy_scenario import (  # noqa
    NumeracyScenario,
    Polynomial,
    RELTYPE_INFO,
    distance_linear,
    distance_parabola,
    distance_plane,
    distance_paraboloid,
)


class DistanceMetric(Metric):
    """
    """

    def evaluate_generation(
        self, adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService
    ) -> List[Stat]:
        """
        """
        references = request_state.instance.references
        _, rel_str, relation_type = map(lambda _: _.output, references)
        input = request_state.instance.input
        datapoint_input = input.split("\n")[-1]
        val = list(map(int, datapoint_input.split(NumeracyScenario.delimiter)))

        distance_func = globals()[f"distance_{relation_type}"]
        result = 0.0
        num_valid = 0
        request_result: RequestResult = request_state.result
        for completion in request_result.completions:
            completion = completion.text.strip()
            try:
                pred = int(completion)
            except Exception:
                continue
            point = val + [pred]
            result += distance_func(point, rel_str)
            num_valid += 1
        percent_valid = 1.0 * num_valid / len(request_result.completions)

        return [
            Stat(MetricName("distance")).add(result),
            Stat(MetricName("percent_valid")).add(percent_valid),
        ]
