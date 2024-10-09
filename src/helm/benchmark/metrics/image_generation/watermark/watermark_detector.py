import os
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torchvision import transforms as T

from helm.benchmark.runner import get_cached_models_path
from helm.common.general import ensure_file_downloaded, hlog
from helm.common.gpu_utils import get_torch_device
from helm.common.images_utils import open_image
from helm.common.optional_dependencies import handle_module_not_found_error


class WatermarkDetector:
    """
    We use LAION's watermark detector (https://github.com/LAION-AI/LAION-5B-WatermarkDetection).
    Adapted from https://github.com/LAION-AI/LAION-5B-WatermarkDetection/blob/main/example_use.py
    """

    MODEL_URL: str = "https://github.com/LAION-AI/LAION-5B-WatermarkDetection/raw/main/models/watermark_model_v1.pt"

    # The example code from LAION used 0.5, but we observed that the watermark detector model could
    # confuse text in an image as a watermark, so we set the threshold to a higher value of 0.9.
    # The detector believes that the test example has a watermark with a 93.563% probability.
    WATERMARK_THRESHOLD: float = 0.9

    @staticmethod
    def load_model():
        """
        Load the watermark detector model.
        """
        try:
            import timm
        except ModuleNotFoundError as e:
            handle_module_not_found_error(e, ["heim"])

        model = timm.create_model("efficientnet_b3a", pretrained=True, num_classes=2)
        model.classifier = nn.Sequential(
            # 1536 is the original in_features
            nn.Linear(in_features=1536, out_features=625),
            nn.ReLU(),  # ReLu to be the activation function
            nn.Dropout(p=0.3),
            nn.Linear(in_features=625, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=2),
        )

        watermark_model_path: str = os.path.join(get_cached_models_path(), "watermark_model_v1.pt")
        ensure_file_downloaded(WatermarkDetector.MODEL_URL, watermark_model_path)
        state_dict = torch.load(watermark_model_path)
        model.load_state_dict(state_dict)
        model.eval()  # Evaluate the model
        return model.to(get_torch_device())

    def __init__(self):
        self._model = self.load_model()

    def has_watermark(self, image_locations: List[str]) -> Tuple[List[bool], List[float]]:
        """
        Returns a list of booleans indicating whether each image (given by `image_locations`)
        contains a watermark or not.
        """
        # Preprocess images (resize and normalize)
        images: List[torch.Tensor] = []
        preprocessing = T.Compose(
            [T.Resize((256, 256)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )
        for location in image_locations:
            # Location can be a file path or a URL
            image = preprocessing(open_image(location).convert("RGB"))
            images.append(image)

        result: List[bool] = []
        probs: List[float] = []
        with torch.no_grad():
            pred = self._model(torch.stack(images).to(get_torch_device()))
            syms = F.softmax(pred, dim=1).detach().cpu().numpy().tolist()
            for i, sym in enumerate(syms):
                watermark_prob, clear_prob = sym
                if watermark_prob > self.WATERMARK_THRESHOLD:
                    hlog(f"Image at {image_locations[i]} has a watermark with {watermark_prob} probability.")
                result.append(watermark_prob >= self.WATERMARK_THRESHOLD)
                probs.append(watermark_prob)
        return result, probs
