from helm.benchmark.window_services.local_window_service import LocalWindowService


class ICEWindowService(LocalWindowService):
    def truncate_from_right(self, text: str, expected_completion_token_length: int = 0) -> str:
        """
        Truncates text from the right to fit within the context window given by `max_request_length`
        minus the expected completion length (defaults to 0).
        """
        max_length: int = self.max_request_length - expected_completion_token_length
        result: str = self.decode(self.encode(text, truncation=True, max_length=max_length).tokens)

        # HACK: For the vast majority of cases, the above logic works, but it sometimes doesn't work
        # for non-English, non-Chinese text (e.g., Japanese text from NaturalQA -
        # followed up in https://github.com/THUDM/icetk/issues/3).
        # Truncate by removing character by character until the prompt fits within the context window.
        while not self.fits_within_context_window(result, expected_completion_token_length):
            result = result[:-1]

        return result
