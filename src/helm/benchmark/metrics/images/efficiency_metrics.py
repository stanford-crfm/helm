from typing import List

from helm.common.request import RequestResult
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.metrics.statistic import Stat
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService


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

        # Models like DALL-E 2 can skip generating images for prompts that violate their content policy
        if not request_result.completions[0].file_location:
            return []

        stats: List[Stat] = [
            Stat(MetricName("prompt_length")).add(len(prompt)),
            Stat(MetricName("inference_runtime")).add(request_result.request_time),
            Stat(MetricName("num_generated_images")).add(len(request_result.completions)),
        ]
        return stats
