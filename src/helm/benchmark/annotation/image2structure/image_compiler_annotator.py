from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from helm.benchmark.annotation.annotator import Annotator, Annotation
from helm.benchmark.adaptation.request_state import RequestState
from helm.common.cache import Cache, CacheConfig
from helm.common.file_caches.local_file_cache import LocalPILFileCache
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.common.media_object import MediaObject

try:
    from PIL import Image
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, suggestions=["images"])


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

    def postprocess_infos(self, infos: Dict[str, Any]) -> Dict[str, Annotation]:
        """Postprocess the infos."""
        return {key: Annotation(value, displayable=False) for key, value in infos.items()}

    def annotate(self, request_state: RequestState) -> List[Dict[str, Annotation]]:
        """Fills the annotations field of the request state with the compiled image."""
        assert request_state.result is not None, "Annotator can only be used after the request has been processed."
        annotations: List[Dict[str, Annotation]] = []
        for completion in request_state.result.completions:
            completion_text: str = completion.text.strip()

            def do_it() -> Dict[str, Annotation]:
                try:
                    assert self._file_cache is not None
                    image, infos = self.compile_completion_into_image(request_state, completion_text)
                    infos = self.postprocess_infos(infos)
                    image_path: str = self._file_cache.store_image(lambda: image)
                    return {
                        "media_object": Annotation(
                            MediaObject(location=image_path, content_type="image/png"), displayable=True
                        ),
                        **infos,
                    }
                except CompilationError as e:
                    return {"error": Annotation(str(e), displayable=True)}

            cache_key: Dict[str, str] = {"completion": completion_text}
            response, _ = self._cache.get(cache_key, do_it)
            annotations.append({**response, "name": Annotation(self.name, displayable=False)})
        return annotations
