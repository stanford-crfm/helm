from typing import Literal

from PIL import Image
from torchmetrics.multimodal import CLIPScore
from torchvision import transforms
import torch

from helm.common.gpu_utils import get_torch_device

_ = torch.manual_seed(42)


class CLIPScorer:
    """
    CLIPScore is a reference free metric that can be used to evaluate the correlation between an image
    caption and the content of the image. It has been found to be highly correlated with human judgement.
    Paper: https://arxiv.org/abs/2104.08718

    We use the TorchMetrics implementation:
    https://torchmetrics.readthedocs.io/en/stable/multimodal/clip_score.html.
    The score is bound between 0 and 100, where a score closer to 100 is better.

    Verified implementation against the scores of image-caption pairs from
    https://wandb.ai/dalle-mini/dalle-mini/reports/OpenAI-CLIP-Score-exploration--VmlldzoxNjMwODM1.
    """

    def __init__(
        self,
        model_name: Literal[
            "openai/clip-vit-base-patch16",
            "openai/clip-vit-base-patch32",
            "openai/clip-vit-large-patch14-336",
            "openai/clip-vit-large-patch14",
        ] = "openai/clip-vit-large-patch14",
    ):
        self._metric = CLIPScore(model_name_or_path=model_name).to(get_torch_device())

    def compute_score(self, caption: str, image_path: str) -> float:
        image: Image = Image.open(image_path)
        image_tensor: torch.Tensor = transforms.ToTensor()(image).to(get_torch_device())
        score: float = self._metric(image_tensor, caption).detach().item()
        return score
