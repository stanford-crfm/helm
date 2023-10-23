from helm.proxy.clients.image_generation.dalle2_client import DALLE2Client
from .clip_window_service import CLIPWindowService
from helm.benchmark.window_services.tokenizer_service import TokenizerService


class DALLE2WindowService(CLIPWindowService):
    def __init__(self, service: TokenizerService):
        super().__init__(service)

    @property
    def max_sequence_length(self) -> int:
        """
        The max sequence length in terms of the number of characters.
        https://beta.openai.com/docs/api-reference/images/create#images/create-prompt
        """
        return DALLE2Client.MAX_PROMPT_LENGTH

    def fits_within_context_window(self, text: str, expected_completion_token_length: int = 0) -> bool:
        return len(text) <= self.max_sequence_length

    def truncate_from_right(self, text: str, expected_completion_token_length: int = 0) -> str:
        return text[: self.max_sequence_length]
