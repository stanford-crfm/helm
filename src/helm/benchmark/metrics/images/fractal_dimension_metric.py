import math
from statistics import mean
from typing import List

from helm.common.request import RequestResult
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.metrics.statistic import Stat
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from .image_metrics_utils import gather_generated_image_locations
from .fractal_dimension.fractal_dimension_util import compute_fractal_dimension


class FractalDimensionMetric(Metric):
    def __repr__(self):
        return "FractalDimensionMetric()"

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.result is not None
        request_result: RequestResult = request_state.result
        image_locations: List[str] = gather_generated_image_locations(request_result)
        if len(image_locations) == 0:
            return []

        fractal_dimensions: List[float] = [
            compute_fractal_dimension(image_location) for image_location in image_locations
        ]
        fractal_dimensions = [dim for dim in fractal_dimensions if not math.isnan(dim)]

        stats: List[Stat] = []
        if len(fractal_dimensions) > 0:
            stats.append(Stat(MetricName("fractal_dimension")).add(mean(fractal_dimensions)))
        return stats
