from typing import List

from helm.common.request import RequestResult
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.metrics.statistic import Stat
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.common.multimodal_request_utils import gather_generated_image_locations


class EfficiencyMetric(Metric):
    """
    Defines the efficiency metrics for text-to-image models.
    """

    def __repr__(self):
        return "EfficiencyMetric()"

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        prompt: str = request_state.request.prompt

        assert request_state.result is not None
        request_result: RequestResult = request_state.result
        image_locations: List[str] = gather_generated_image_locations(request_result)
        if len(image_locations) == 0:
            return []

        # inference_runtime is computed in BasicMetric
        stats: List[Stat] = [
            Stat(MetricName("prompt_length")).add(len(prompt)),
            Stat(MetricName("num_generated_images")).add(len(request_result.completions)),
        ]
        return stats
