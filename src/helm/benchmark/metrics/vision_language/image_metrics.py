from typing import List, Tuple, Dict
import numpy as np
from abc import ABC, abstractmethod

from helm.benchmark.metrics.evaluate_instances_metric import EvaluateInstancesMetric
from helm.common.request import RequestResult
from helm.benchmark.adaptation.request_state import RequestState
from helm.common.optional_dependencies import handle_module_not_found_error
from ..metric_name import MetricName
from ..statistic import Stat
from .image_utils import preprocess_image, earth_movers_distance, pixel_similarity, sift_similarity

try:
    from PIL import Image
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, suggestions=["images"])


class CompilationError(Exception):
    pass


class ImageMetric(EvaluateInstancesMetric, ABC):
    COMPILE_METRIC: str = "compile"
    EARTH_MOVERS_SIMILARITY: str = "earth-movers"
    PIXEL_SIMILARITY: str = "pixel"
    SIFT_SIMILARITY: str = "sift"

    def __init__(self, metric_names: List[str]):
        self._metric_names: List[str] = metric_names

    @abstractmethod
    def compile_completion_into_image(self, request_state: RequestState, completion: str, ref_image: Image) -> Image:
        raise NotImplementedError

    def evaluate_instances(self, request_states: List[RequestState]) -> List[Stat]:
        stats_dict: Dict[str, Stat] = {
            name: Stat(MetricName(name)) for name in (self._metric_names + [self.COMPILE_METRIC])
        }

        for request_state in request_states:
            reference = request_state.instance.references[0]
            assert reference.output.multimedia_content is not None
            assert reference.output.multimedia_content.type == "image"
            ref_image: Image
            rgb_ref_image: np.ndarray
            gray_ref_image: np.ndarray
            if reference.output.multimedia_content.is_local_file:
                ref_image: Image = Image.open(reference.output.multimedia_content.location)
                rgb_ref_image = np.array(ref_image)
                gray_ref_image = preprocess_image(ref_image)
            else:
                raise Exception(
                    "Remote images are not supported in metrics. "
                    "Images should be downloaded when constructing the instance."
                )

            assert request_state.result is not None
            request_result: RequestResult = request_state.result
            # Filter out empty completions
            completions: List[str] = [
                completion.text.strip() for completion in request_result.completions if completion.text
            ]

            for completion in completions:
                image: Image
                try:
                    image = self.compile_completion_into_image(request_state, completion, ref_image)
                except CompilationError:
                    stats_dict[self.COMPILE_METRIC].add(0)  # Did not compile
                    # For all other metrics, we set the value to zero
                    for metric_name in self._metric_names:
                        stats_dict[metric_name].add(0)
                    continue
                assert image.size == ref_image.size
                rgb_image: np.ndarray = np.array(image)
                gray_image: np.ndarray = preprocess_image(image)

                if self.PIXEL_SIMILARITY in self._metric_names:
                    stats_dict[self.PIXEL_SIMILARITY].add(pixel_similarity(gray_image, gray_ref_image))
                if self.SIFT_SIMILARITY in self._metric_names:
                    stats_dict[self.SIFT_SIMILARITY].add(sift_similarity(rgb_image, rgb_ref_image))
                if self.EARTH_MOVERS_SIMILARITY in self._metric_names:
                    stats_dict[self.EARTH_MOVERS_SIMILARITY].add(
                        1.0 - earth_movers_distance(gray_image, gray_ref_image)
                    )

        return list(stats_dict.values())
