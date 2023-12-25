from abc import ABC, abstractmethod
from urllib.request import urlretrieve
import os

import torch

from helm.common.general import ensure_directory_exists
from helm.common.hierarchical_logger import hlog
from helm.common.gpu_utils import get_torch_device
from helm.common.images_utils import open_image
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.benchmark.runner import get_cached_models_path


class AestheticsScorer(ABC):
    """
    Abstract class for aesthetics scorers.
    """

    @abstractmethod
    def compute_aesthetics_score(self, image_location: str) -> float:
        """
        Compute the aesthetics score. Returns a value between 1 and 10.
        """
        pass


class AestheticsScorerV1(AestheticsScorer):
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
        cache_folder: str = os.path.join(get_cached_models_path(), "emb_reader")
        ensure_directory_exists(cache_folder)
        model_path: str = os.path.join(cache_folder, f"sa_0_4_{clip_model}_linear.pth")

        if not os.path.exists(model_path):
            model_url: str = os.path.join(AestheticsScorerV1.MODEL_URL_TEMPLATE.format(clip_model=clip_model))
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
        try:
            import clip
        except ModuleNotFoundError as e:
            handle_module_not_found_error(e, ["heim"])

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


class AestheticsScorerV2(AestheticsScorer):
    """
    LAION's CLIP-based Aesthetics Predictor v2: https://laion.ai/blog/laion-aesthetics

    Adapted from
    https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/main/simple_inference.py
    """

    MODEL_URL: str = (
        "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14-linearMSE.pth"
    )

    def __init__(self):
        """Load the aesthetics model."""
        import pytorch_lightning as pl
        import torch.nn as nn
        import torch.nn.functional as F
        import clip

        class MLP(pl.LightningModule):
            def __init__(self, input_size, xcol="emb", ycol="avg_rating"):
                super().__init__()
                self.input_size = input_size
                self.xcol = xcol
                self.ycol = ycol
                self.layers = nn.Sequential(
                    nn.Linear(self.input_size, 1024),
                    # nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(1024, 128),
                    # nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 64),
                    # nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(64, 16),
                    # nn.ReLU(),
                    nn.Linear(16, 1),
                )

            def forward(self, x):
                return self.layers(x)

            def training_step(self, batch, batch_idx):
                x = batch[self.xcol]
                y = batch[self.ycol].reshape(-1, 1)
                x_hat = self.layers(x)
                loss = F.mse_loss(x_hat, y)
                return loss

            def validation_step(self, batch, batch_idx):
                x = batch[self.xcol]
                y = batch[self.ycol].reshape(-1, 1)
                x_hat = self.layers(x)
                loss = F.mse_loss(x_hat, y)
                return loss

            def configure_optimizers(self):
                optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
                return optimizer

        # Load the CLIP and aesthetics model
        self._device: torch.device = get_torch_device()

        self._model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14

        cache_folder: str = os.path.join(get_cached_models_path(), "aesthetics_v2")
        ensure_directory_exists(cache_folder)
        model_path: str = os.path.join(cache_folder, "sac+logos+ava1-l14-linearMSE.pth")
        if not os.path.exists(model_path):
            hlog(f"Downloading the model to {model_path}...")
            urlretrieve(self.MODEL_URL, model_path)

        s = torch.load(model_path)
        self._model.load_state_dict(s)
        self._model.to(self._device)
        self._model.eval()

        self._model2, self._preprocess = clip.load("ViT-L/14", device=self._device)  # RN50x64

    def compute_aesthetics_score(self, image_location: str) -> float:
        """
        Compute the aesthetics score. Returns a value between 1 and 10.
        """
        import numpy as np

        def normalized(a, axis=-1, order=2):
            l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
            l2[l2 == 0] = 1
            return a / np.expand_dims(l2, axis)

        pil_image = open_image(image_location)
        image = self._preprocess(pil_image).unsqueeze(0).to(self._device)
        with torch.no_grad():
            image_features = self._model2.encode_image(image)

        im_emb_arr = normalized(image_features.cpu().detach().numpy())
        prediction = self._model(
            torch.from_numpy(im_emb_arr).to(self._device).type(torch.cuda.FloatTensor)  # type: ignore
        )
        hlog(f"location: {image_location} score: {float(prediction)}")
        return float(prediction)
