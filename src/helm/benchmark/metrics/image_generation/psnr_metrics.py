from typing import List

from torchvision import transforms
import torch

from helm.common.gpu_utils import get_torch_device
from helm.common.images_utils import open_image
from helm.common.request import RequestResult
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.metrics.statistic import Stat
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.common.multimodal_request_utils import gather_generated_image_locations, get_gold_image_location


class PeakSignalToNoiseRatioMetric(Metric):
    """
    Peak signal-to-noise ratio (PSNR) is the ratio between the maximum possible power of
    a signal and the power of corrupting noise that affects the fidelity of its representation.

    We use the TorchMetrics implementation:
    https://torchmetrics.readthedocs.io/en/stable/image/peak_signal_noise_ratio.html
    """

    def __init__(self):
        self._metric = None
        self._device = get_torch_device()

    def __repr__(self):
        return "PeakSignalToNoiseRatioMetric()"

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
        score: float = self._compute_psnr_scores(image_locations, gold_image_path)
        return [Stat(MetricName("expected_psnr_score")).add(score)]

    def _compute_psnr_scores(self, generated_image_locations: List[str], reference_image_path: str) -> float:
        try:
            from torchmetrics import PeakSignalNoiseRatio
        except ModuleNotFoundError as e:
            handle_module_not_found_error(e, ["heim"])

        if self._metric is None:
            self._metric = PeakSignalNoiseRatio().to(self._device)

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
