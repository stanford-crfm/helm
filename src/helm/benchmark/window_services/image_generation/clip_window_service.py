from abc import ABC

from helm.benchmark.window_services.local_window_service import LocalWindowService


class CLIPWindowService(LocalWindowService, ABC):
    def truncate_from_right(self, text: str, expected_completion_token_length: int = 0) -> str:
        result: str = self.decode(self.encode(text, truncation=True, max_length=self.max_request_length).tokens)

        # HACK: For the vast majority of cases, the above logic works, but there are a few where the
        # token count exceeds `max_length` by 1.
        while not self.fits_within_context_window(result):
            result = result[:-1]

        return result
