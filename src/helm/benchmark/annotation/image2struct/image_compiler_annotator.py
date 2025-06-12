from abc import ABC, abstractmethod
from threading import Lock
from typing import Any, Dict, List, Tuple, Callable

from dacite import from_dict

from helm.benchmark.annotation.annotator import Annotator
from helm.benchmark.adaptation.request_state import RequestState
from helm.common.cache import Cache, CacheConfig
from helm.common.file_caches.local_file_cache import LocalPILFileCache
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.common.media_object import MediaObject
from helm.proxy.retry import get_retry_decorator

try:
    from PIL import Image
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, suggestions=["images"])


_compilation_lock = Lock()


def retry_if_compilation_failed(result: Dict[str, Any]) -> bool:
    """Retries when the compilation fails."""
    return "unknown_error" in result


retry: Callable = get_retry_decorator(
    "Compilation", max_attempts=5, wait_exponential_multiplier_seconds=2, retry_on_result=retry_if_compilation_failed
)


class CompilationError(Exception):
    pass


class ImageCompilerAnnotator(Annotator, ABC):
    """Annotator that compiles the text completions into an image."""

    def __init__(self, cache_config: CacheConfig, file_storage_path: str):
        self._cache = Cache(cache_config)
        self._file_cache = LocalPILFileCache(file_storage_path)

    @abstractmethod
    def compile_completion_into_image(
        self, request_state: RequestState, completion_text: str
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        raise NotImplementedError

    def postprocess_infos(self, infos: Dict[str, Any]) -> Dict[str, Any]:
        """Postprocess the infos."""
        return infos

    def annotate(self, request_state: RequestState) -> List[Dict[str, Any]]:
        """Fills the annotations field of the request state with the compiled image."""
        assert request_state.result is not None, "Annotator can only be used after the request has been processed."
        annotations: List[Dict[str, Any]] = []
        for completion in request_state.result.completions:
            completion_text: str = completion.text.strip()
            raw_response: Dict[str, Any]

            @retry
            def compile() -> Dict[str, Any]:
                def do_it() -> Dict[str, Any]:
                    try:
                        assert self._file_cache is not None
                        image, infos = self.compile_completion_into_image(request_state, completion_text)
                        infos = self.postprocess_infos(infos)
                        image_path: str = self._file_cache.store_image(lambda: image)
                        return {
                            "media_object": MediaObject(location=image_path, content_type="image/png").to_dict(),
                            **infos,
                        }
                    except CompilationError as e:
                        return {"error": str(e)}

                try:
                    cache_key: Dict[str, str] = {"completion": completion_text}
                    raw_response, _ = self._cache.get(cache_key, do_it)
                    return raw_response
                except Exception as e:
                    return {"unknown_error": str(e)}

            with _compilation_lock:
                raw_response = compile()
            response = {**raw_response}
            if "media_object" in response:
                response["media_object"] = from_dict(MediaObject, response["media_object"])

            # Merge annotations
            annotations.append(response)
        return annotations
