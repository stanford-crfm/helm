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
from helm.common.multimodal_request_utils import gather_generated_image_locations
from helm.benchmark.metrics.image_generation.fractal_dimension.fractal_dimension_util import compute_fractal_dimension


class FractalDimensionMetric(Metric):

    # From https://www.nature.com/articles/35065154, "participants in the perception study consistently
    # preferred fractals with D values in the range of 1.3 to 1.5, irrespective of the pattern's origin.
    # Significantly, many of the fractal patterns surrounding us in nature have D values in this range.
    # Clouds have a value of 1.3."
    IDEAL_FRACTAL_DIMENSION: float = 1.4

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
        fractal_dimension_losses: List[float] = [
            abs(dim - self.IDEAL_FRACTAL_DIMENSION) for dim in fractal_dimensions if not math.isnan(dim)
        ]

        stats: List[Stat] = []
        if len(fractal_dimension_losses) > 0:
            stats.append(Stat(MetricName("fractal_dimension_loss")).add(mean(fractal_dimension_losses)))
        return stats
