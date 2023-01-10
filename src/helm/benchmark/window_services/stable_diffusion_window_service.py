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

    def truncate_from_right(self, text: str, expected_completion_token_length: int = 0) -> str:
        max_length: int = self.max_request_length
        result: str = self.decode(self.encode(text, truncation=True, max_length=max_length).tokens)

        # HACK: For the vast majority of cases, the above logic works, but there are a few where
        #       the token count exceeds `max_length` by 1.
        while not self.fits_within_context_window(result):
            result = result[:-1]
        return result
