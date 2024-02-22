from typing import Any, Dict, Optional
import os

from helm.common.cache import Cache, CacheConfig
from helm.common.general import ensure_file_downloaded, ensure_directory_exists
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.common.nudity_check_request import NudityCheckRequest, NudityCheckResult


class NudityCheckClientError(Exception):
    pass


class NudityCheckClient:
    MODEL_DOWNLOAD_URL: str = (
        "https://worksheets.codalab.org/rest/bundles/0x81f7d1febb374fdcad6d9ae970f46750/contents/blob/"
    )

    def __init__(self, cache_config: CacheConfig):
        try:
            from nudenet import NudeClassifier
        except ModuleNotFoundError as e:
            handle_module_not_found_error(e, ["heim"])

        self.cache = Cache(cache_config)
        self._nudity_classifier: Optional[NudeClassifier] = None

    def check_nudity(self, request: NudityCheckRequest) -> NudityCheckResult:
        """Check for nudity for a batch of images using NudeNet."""
        try:
            from nudenet import NudeClassifier
        except ModuleNotFoundError as e:
            handle_module_not_found_error(e, ["heim"])

        try:

            def do_it() -> Dict[str, Any]:
                if self._nudity_classifier is None:
                    # The NudeNet library does not automatically provide model weights that work, so
                    # manually download them. The path is hardcoded in the NudeNet library.
                    base_path: str = os.path.join(os.path.expanduser("~"), ".NudeNet")
                    ensure_directory_exists(base_path)
                    model_path: str = os.path.join(base_path, "classifier_model.onnx")
                    ensure_file_downloaded(source_url=self.MODEL_DOWNLOAD_URL, target_path=model_path)
                    self._nudity_classifier = NudeClassifier()

                path_to_nudity_scores: Dict[str, Dict[str, float]] = self._nudity_classifier.classify(
                    request.image_locations
                )
                return path_to_nudity_scores

            results, cached = self.cache.get({"locations": sorted(request.image_locations)}, do_it)
        except Exception as e:
            raise NudityCheckClientError(e)

        nudity_results: Dict[str, bool] = {
            image_location: nudity_result["unsafe"] > nudity_result["safe"]
            for image_location, nudity_result in results.items()
        }
        return NudityCheckResult(
            success=True,
            cached=cached,
            image_to_nudity=nudity_results,
        )
