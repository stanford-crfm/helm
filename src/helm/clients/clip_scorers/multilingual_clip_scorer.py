import torch
import transformers

from helm.common.gpu_utils import get_torch_device, get_torch_device_name
from helm.common.images_utils import open_image
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.clients.clip_scorers.base_clip_scorer import BaseCLIPScorer

_ = torch.manual_seed(42)


class MultilingualCLIPScorer(BaseCLIPScorer):
    """
    Multilingual-CLIP extends OpenAI's English text encoders to multiple other languages.
    Adapted from https://huggingface.co/M-CLIP/XLM-Roberta-Large-Vit-L-14
    """

    TEXT_MODEL_NAME: str = "M-CLIP/XLM-Roberta-Large-Vit-L-14"
    IMAGE_MODEL_NAME: str = "ViT-L/14"

    def __init__(self):
        try:
            import clip
            from multilingual_clip import pt_multilingual_clip
        except ModuleNotFoundError as e:
            handle_module_not_found_error(e, ["heim"])

        super().__init__()
        self._device: torch.device = get_torch_device()
        self._text_model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(self.TEXT_MODEL_NAME)
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(self.TEXT_MODEL_NAME)
        self._model, self._preprocess = clip.load(self.IMAGE_MODEL_NAME, device=get_torch_device_name())

    def compute_score(self, caption: str, image_location: str) -> float:
        # Get text features
        text_features = self._text_model.forward(caption, self._tokenizer)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        text_features = text_features.to(self._device)

        image = open_image(image_location)
        image = self._preprocess(image).unsqueeze(0).to(self._device)

        # Get image features
        with torch.no_grad():
            image_features = self._model.encode_image(image)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

        # Compute score using text and image features
        score = 100 * (image_features * text_features).sum(axis=-1)
        return score.detach().item()
