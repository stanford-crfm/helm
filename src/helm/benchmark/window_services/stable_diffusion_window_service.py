from .clip_window_service import CLIPWindowService
from .tokenizer_service import TokenizerService


class StableDiffusionWindowService(CLIPWindowService):
    def __init__(self, service: TokenizerService):
        super().__init__(service)

    @property
    def max_sequence_length(self) -> int:
        """
        From https://huggingface.co/blog/stable_diffusion, "...the text prompt is transformed to text
        embeddings of size 77×768 via CLIP's text encoder."
        """
        return 77
