from typing import List

from torchvision import transforms
import torch

from helm.common.gpu_utils import get_torch_device
from helm.common.images_utils import open_image
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.common.request import RequestResult
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.metrics.statistic import Stat
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.common.multimodal_request_utils import gather_generated_image_locations, get_gold_image_location


class MultiScaleStructuralSimilarityIndexMeasureMetric(Metric):
    """
    The Multi-scale Structural Similarity Index Measure (MS-SSIM) is measure of image quality and
    a generalization of Structural Similarity Index Measure (SSIM) by incorporating image details
    at different resolution scores. The SSIM is a method for predicting the perceived quality of
    digital television and cinematic pictures, as well as other kinds of digital images and videos.
    SSIM is used for measuring the similarity between two images.

    We use the TorchMetrics implementation:
    https://torchmetrics.readthedocs.io/en/stable/image/multi_scale_structural_similarity.html
    """

    def __init__(self):
        self._metric = None
        self._device = get_torch_device()

    def __repr__(self):
        return "MultiScaleStructuralSimilarityIndexMeasureMetric()"

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
        score: float = self._compute_ssim_scores(image_locations, gold_image_path)
        return [Stat(MetricName("expected_multi_scale_ssim_score")).add(score)]

    def _compute_ssim_scores(self, generated_image_locations: List[str], reference_image_path: str) -> float:
        try:
            from torchmetrics import MultiScaleStructuralSimilarityIndexMeasure
        except ModuleNotFoundError as e:
            handle_module_not_found_error(e, ["heim"])

        if self._metric is None:
            self._metric = MultiScaleStructuralSimilarityIndexMeasure().to(self._device)

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
