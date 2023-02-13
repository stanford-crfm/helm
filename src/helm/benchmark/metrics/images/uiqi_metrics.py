from typing import List
import math

from torchmetrics import UniversalImageQualityIndex
from torchvision import transforms
import torch

from helm.common.gpu_utils import get_torch_device
from helm.common.images_utils import open_image
from helm.common.request import RequestResult
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.metrics.statistic import Stat
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from .image_metrics_util import gather_generated_image_locations, get_gold_image_location


class UniversalImageQualityIndexMetric(Metric):
    """
    Universal Image Quality Index (UIQI) from https://ieeexplore.ieee.org/document/995823.
    The range of UIQI is [-1, 1].

    We use the TorchMetrics implementation:
    https://torchmetrics.readthedocs.io/en/stable/image/universal_image_quality_index.html
    """

    def __init__(self):
        self._metric = None
        self._device = get_torch_device()

    def __repr__(self):
        return "UniversalImageQualityIndexMetric()"

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

        gold_image_path: str = get_gold_image_location(request_state)
        score: float = self._compute_uiqi_scores(image_locations, gold_image_path)
        if math.isnan(score) or score == -math.inf or score == math.inf:
            return []
        return [Stat(MetricName("expected_uiqi_score")).add(score)]

    def _compute_uiqi_scores(self, generated_image_locations: List[str], reference_image_path: str) -> float:
        if self._metric is None:
            self._metric = UniversalImageQualityIndex().to(self._device)

        preprocessing = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ]
        )
        generated_images: List[torch.Tensor] = []
        reference_images: List[torch.Tensor] = []
        for location in generated_image_locations:
            image = preprocessing(open_image(location))
            generated_images.append(image)
            image = preprocessing(open_image(reference_image_path))
            reference_images.append(image)

        img1: torch.Tensor = torch.stack(generated_images).to(self._device)
        img2: torch.Tensor = torch.stack(reference_images).to(self._device)
        score: float = self._metric(img1, img2).detach().item()
        return score
