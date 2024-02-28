from typing import Optional

from helm.common.cache import Cache, SqliteCacheConfig
from helm.common.optional_dependencies import handle_module_not_found_error

try:
    from google.cloud import translate_v2 as translate  # type: ignore
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, ["heim"])


class GoogleTranslateClient:
    """
    Client for Google Translate.
    Follow the instructions at https://cloud.google.com/translate/docs/setup to use this client.

    # TODO: add this as a central service
    """

    def __init__(self, cache_path: str = "prod_env/cache/google_translate.sqlite"):
        self.translate_client: Optional[translate.Client] = None
        self.cache = Cache(SqliteCacheConfig(cache_path))

    def translate(self, text: str, target_language: str) -> str:
        def do_it():
            if self.translate_client is None:
                self.translate_client = translate.Client()

            result = self.translate_client.translate(text, target_language=target_language)
            del result["input"]
            assert "translatedText" in result, f"Invalid response: {result}"
            return result

        response, _ = self.cache.get({"text": text, "target_language": target_language}, do_it)
        return response["translatedText"]
