from typing import List

from helm.benchmark.scenarios.scenario import Reference
from helm.common.gpu_utils import get_torch_device
from helm.common.images_utils import open_image
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision import transforms
import torch

from helm.common.request import RequestResult
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.metrics.statistic import Stat
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService


class LPIPSMetric(Metric):
    """
    The Learned Perceptual Image Patch Similarity (LPIPS) is used to judge the perceptual similarity between
    two images. LPIPS essentially computes the similarity between the activations of two image patches for
    some pre-defined network. This measure has been shown to match human perception well. A low LPIPS score
    means that image patches are perceptual similar.

    We use the TorchMetrics implementation:
    https://torchmetrics.readthedocs.io/en/stable/image/learned_perceptual_image_patch_similarity.html.
    """

    def __init__(self):
        self._metric = None
        self._device = get_torch_device()

    def __repr__(self):
        return "LPIPSMetric()"

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.result is not None
        request_result: RequestResult = request_state.result

        # Gather the image locations
        image_locations: List[str] = []
        for image in request_result.completions:
            # Models like DALL-E 2 can skip generating images for prompts that violate their content policy
            if image.file_location is None:
                return []

            image_locations.append(image.file_location)

        # Get the gold image
        references: List[Reference] = request_state.instance.references
        assert len(references) > 0 and references[0].output.file_path is not None, "Need a gold image to compute LPIPS"
        gold_image_path: str = references[0].output.file_path

        # Batch process the images and compute the average LPIPS score.
        score: float = self._compute_lpips_scores(image_locations, gold_image_path)
        return [Stat(MetricName("expected_lpips_score")).add(score)]

    def _compute_lpips_scores(self, generated_image_locations: List[str], reference_image_path: str) -> float:
        if self._metric is None:
            self._metric = LearnedPerceptualImagePatchSimilarity(net_type="vgg").to(self._device)

        preprocessing = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
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
