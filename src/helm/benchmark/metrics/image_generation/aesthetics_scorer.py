from urllib.request import urlretrieve
import os

import torch

from helm.common.general import ensure_directory_exists, get_helm_cache_path
from helm.common.gpu_utils import get_torch_device
from helm.common.images_utils import open_image


class AestheticsScorer:
    """
    LAION's CLIP-based aesthetics predictor for images (https://github.com/LAION-AI/aesthetic-predictor).
    Adapted from
    https://colab.research.google.com/github/LAION-AI/aesthetic-predictor/blob/main/asthetics_predictor.ipynb.
    """

    MODEL_URL_TEMPLATE: str = (
        "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_{clip_model}_linear.pth?raw=true"
    )

    @staticmethod
    def load_model(clip_model="vit_l_14"):
        """Load the aesthetics model."""
        cache_folder: str = os.path.join(get_helm_cache_path(), "emb_reader")
        ensure_directory_exists(cache_folder)
        model_path: str = os.path.join(cache_folder, f"sa_0_4_{clip_model}_linear.pth")

        if not os.path.exists(model_path):
            model_url: str = os.path.join(AestheticsScorer.MODEL_URL_TEMPLATE.format(clip_model=clip_model))
            urlretrieve(model_url, model_path)

        if clip_model == "vit_l_14":
            m = torch.nn.Linear(768, 1)
        elif clip_model == "vit_b_32":
            m = torch.nn.Linear(512, 1)
        else:
            raise ValueError(f"Invalid model: {clip_model}")

        s = torch.load(model_path)
        m.load_state_dict(s)
        m.eval()
        return m

    def __init__(self):
        import clip

        # Load the CLIP and aesthetics model
        self._device: torch.device = get_torch_device()
        self._model, self._preprocess = clip.load("ViT-L/14", device=self._device)
        self._aesthetics_model = self.load_model().to(self._device)

    def compute_aesthetics_score(self, image_location: str) -> float:
        """
        Compute the aesthetics score. Returns a value between 1 and 10.
        """
        image = self._preprocess(open_image(image_location)).unsqueeze(0).to(self._device)
        with torch.no_grad():
            image_features = self._model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            return self._aesthetics_model(image_features.float()).detach().item()
