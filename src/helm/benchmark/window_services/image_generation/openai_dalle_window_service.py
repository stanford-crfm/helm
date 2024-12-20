from helm.benchmark.window_services.image_generation.clip_window_service import CLIPWindowService


class OpenAIDALLEWindowService(CLIPWindowService):
    def fits_within_context_window(self, text: str, expected_completion_token_length: int = 0) -> bool:
        return len(text) <= self.max_sequence_length

    def truncate_from_right(self, text: str, expected_completion_token_length: int = 0) -> str:
        return text[: self.max_sequence_length]
