from typing import List, Optional
import os
import pickle

import torch

from helm.common.gpu_utils import get_torch_device, is_cuda_available
from helm.common.optional_dependencies import handle_module_not_found_error


class Q16ToxicityDetector:
    """
    From https://arxiv.org/abs/2202.06675, Q16 is a CLIP-based toxicity detector for images.
    Adapted from https://colab.research.google.com/drive/1_JoaTV-E83ejcjcT4FlMG0_lg_8YODPZ?usp=sharing
    """

    # label 1 corresponds to inappropriate material
    CORRECT_LABEL: int = 1

    def __init__(self):
        # Load CLIP and the classifier
        self._is_cuda_available: bool = is_cuda_available()
        self._device: torch.device = get_torch_device()
        self._clip_wrapper: Optional[ClipWrapper] = None
        self._classifier: Optional[SimClassifier] = None

    def _load_models(self):
        def load_prompts():
            base_path: str = os.path.dirname(__file__)
            prompts_path: str = os.path.join(base_path, "prompts.p")
            file = pickle.load(open(prompts_path, "rb"))
            tensor = torch.HalfTensor(file) if self._is_cuda_available else torch.Tensor(file)
            return tensor.to(self._device)

        if self._clip_wrapper is None or self._classifier is None:
            self._clip_wrapper = ClipWrapper(self._device)
            trained_prompts = load_prompts()
            self._classifier = SimClassifier(trained_prompts)

    def _compute_embeddings(self, image_paths: List[str]):
        try:
            from PIL import Image
        except ModuleNotFoundError as e:
            handle_module_not_found_error(e, ["heim"])

        assert self._clip_wrapper is not None
        images = [self._clip_wrapper.preprocess(Image.open(image_path)) for image_path in image_paths]  # type: ignore
        image_tensors: torch.Tensor = torch.stack(images).to(self._device)
        embeddings = self._clip_wrapper(image_tensors)
        return embeddings.half() if self._is_cuda_available else embeddings

    def is_inappropriate(self, image_path: str) -> bool:
        """
        Returns a boolean indicating whether the image is appropriate or not.
        """
        self._load_models()
        assert self._classifier is not None
        embeddings = self._compute_embeddings([image_path])
        y = self._classifier(embeddings)
        label: float = torch.argmax(y, dim=0).item()
        return label == self.CORRECT_LABEL


class ClipWrapper(torch.nn.Module):
    def __init__(self, device: torch.device, model_name="ViT-L/14"):
        try:
            import clip
        except ModuleNotFoundError as e:
            handle_module_not_found_error(e, ["heim"])

        super(ClipWrapper, self).__init__()
        self.clip_model, self.preprocess = clip.load(model_name, device, jit=False)
        self.clip_model.eval()

    def forward(self, x: torch.Tensor):
        return self.clip_model.encode_image(x)


class SimClassifier(torch.nn.Module):
    def __init__(self, embeddings):
        super(SimClassifier, self).__init__()
        self.embeddings = torch.nn.parameter.Parameter(embeddings)

    def forward(self, x):
        embeddings_norm = self.embeddings / self.embeddings.norm(dim=-1, keepdim=True)
        # Pick the top 5 most similar labels for the image
        image_features_norm = x / x.norm(dim=-1, keepdim=True)

        similarity = 100.0 * image_features_norm @ embeddings_norm.T
        return similarity.squeeze()
