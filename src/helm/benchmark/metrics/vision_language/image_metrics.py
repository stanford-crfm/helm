from typing import List, Dict, Optional
import numpy as np
from torchvision import transforms
from abc import ABC, abstractmethod
import os

from helm.benchmark.metrics.evaluate_instances_metric import EvaluateInstancesMetric
from helm.common.images_utils import open_image
from helm.common.gpu_utils import get_torch_device
from helm.common.file_caches.local_file_cache import LocalPILFileCache
from helm.common.cache import Cache, SqliteCacheConfig
from helm.common.request import RequestResult
from helm.benchmark.adaptation.request_state import RequestState
from helm.common.media_object import MediaObject
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.common.hierarchical_logger import hlog
from ..metric_name import MetricName
from ..statistic import Stat
from .image_utils import preprocess_image, earth_mover_similarity, pixel_similarity, sift_similarity

try:
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    from PIL import Image
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, suggestions=["image2structure"])


class CompilationError(Exception):
    pass


class GenerateImageFromCompletionMetric(EvaluateInstancesMetric, ABC):
    """Abstract class for image metrics.

    This class is designed to evaluate metrics on images that should be generated using the text
    ouput of the model, such as LaTeX, HTML, etc.

    The class provides a method to compile the completion into an image and then evaluate the
    similarity between the generated image and the reference image using different metrics.

    In addition to the metrics, the class also provides a metric to evaluate the compilation success.
    If the compilation fails, the similarity metrics are not evaluated and are all set to the most
    dissimilar value.

    Since compilation can be expensive, the class provides a cache to store the compiled images.
    In addition metrics can also be cached to avoid recomputation.
    """

    # Metric names
    COMPILE_METRIC: str = "compilation_success"
    EARTH_MOVER_SIMILARITY: str = "earth_mover_similarity"
    PIXEL_SIMILARITY: str = "pixel_similarity"
    SIFT_SIMILARITY: str = "sift_similarity"
    LPIPS_SIMILARITY: str = "lpips_similarity"

    def __init__(self, generation_type: str, metric_names: List[str], normalize_by_white_score: bool = False):
        self.generation_type = generation_type
        self._metric_names: List[str] = metric_names
        self._lpips_metric: Optional[LearnedPerceptualImagePatchSimilarity] = None
        self._device = get_torch_device()
        self._normalize_by_white_score = normalize_by_white_score
        self._cache: Optional[Cache] = None
        self._file_cache: Optional[LocalPILFileCache] = None

    def _get_file_cache(self, file_storage_path: str) -> LocalPILFileCache:
        # Initialize `FileCache` to store compiled images
        local_file_cache_path: str = os.path.join(file_storage_path, self.generation_type)
        return LocalPILFileCache(local_file_cache_path)

    def _get_cache(self, path: str) -> Cache:
        # Initialize `Cache` to store compiled images
        sql_cache_path: str = os.path.join(path, f"{self.generation_type}.sqlite")
        return Cache(SqliteCacheConfig(sql_cache_path))

    def _get_compilation_cache_key(self, completion: str) -> Dict[str, str]:
        return {
            "generation_type": self.generation_type,
            "completion": completion,
        }

    @abstractmethod
    def compile_completion_into_image(
        self, request_state: RequestState, completion: str, ref_image: Image.Image
    ) -> Image.Image:
        raise NotImplementedError

    def evaluate_instances(self, request_states: List[RequestState], eval_cache_path: str) -> List[Stat]:
        if self._cache is None:
            self._cache = self._get_cache(eval_cache_path)
        if self._file_cache is None:
            self._file_cache = self._get_file_cache(eval_cache_path)

        stats_dict: Dict[str, Stat] = {
            name: Stat(MetricName(name)) for name in (self._metric_names + [self.COMPILE_METRIC])
        }

        for request_state in request_states:
            reference = request_state.instance.references[0]
            assert reference.output.multimedia_content is not None
            assert len(reference.output.multimedia_content.media_objects) > 0
            ref_media_object: MediaObject = reference.output.multimedia_content.media_objects[0]
            assert ref_media_object.type == "image"
            ref_image: Image.Image
            rgb_ref_image: np.ndarray
            gray_ref_image: np.ndarray
            if ref_media_object.is_local_file and ref_media_object.location is not None:
                ref_image = open_image(ref_media_object.location)
                rgb_ref_image = np.array(ref_image)
                gray_ref_image = preprocess_image(ref_image)
            else:
                raise Exception(
                    "Remote images are not supported in metrics. "
                    "Images should be downloaded when constructing the instance."
                )
            white_image: Optional[Image.Image] = None
            rgb_white_image: Optional[np.ndarray] = None
            gray_white_image: Optional[np.ndarray] = None
            if self._normalize_by_white_score:
                white_image = Image.new("RGB", ref_image.size, (255, 255, 255))
                rgb_white_image = np.array(white_image)
                gray_white_image = preprocess_image(white_image)

            assert request_state.result is not None
            request_result: RequestResult = request_state.result
            # Filter out empty completions
            completions: List[str] = [
                completion.text.strip() for completion in request_result.completions if completion.text
            ]

            for completion in completions:

                def do_it():
                    try:
                        assert self._file_cache is not None
                        image_path: str = self._file_cache.store_image(
                            lambda: self.compile_completion_into_image(request_state, completion, ref_image)
                        )
                        return {"image_path": image_path}
                    except CompilationError as e:
                        return {"error": str(e)}

                cache_key = self._get_compilation_cache_key(completion)
                response, _ = self._cache.get(cache_key, do_it)

                if "error" in response:
                    stats_dict[self.COMPILE_METRIC].add(0)  # Did not compile
                    # For all other metrics, we set the value to zero
                    for metric_name in self._metric_names:
                        stats_dict[metric_name].add(0)
                    continue

                image: Image.Image = self._file_cache.load_image(response["image_path"])
                assert image.size == ref_image.size
                rgb_image: np.ndarray = np.array(image)
                gray_image: np.ndarray = preprocess_image(image)

                # List of metrics and arguments to evaluate
                # The arguments are as follows:
                # 1. Name of the metric
                # 2. Function to compute the metric
                # 3. The generated image obhect (type can depend on the metric)
                # 4. The reference image object (type can depend on the metric)
                # 5. The white image object (type can depend on the metric) - may be used to normalize the metric
                # 6. Whether the metric can be computed on the white image - if not, the metric is not normalized
                metric_runs: list = [
                    [self.PIXEL_SIMILARITY, pixel_similarity, gray_image, gray_ref_image, gray_white_image, True],
                    [self.SIFT_SIMILARITY, sift_similarity, rgb_image, rgb_ref_image, rgb_white_image, False],
                    [
                        self.EARTH_MOVER_SIMILARITY,
                        earth_mover_similarity,
                        gray_image,
                        gray_ref_image,
                        gray_white_image,
                        True,
                    ],
                    [self.LPIPS_SIMILARITY, self.lpips_similarity, image, ref_image, white_image, True],
                ]

                for metric_name, metric_fn, image1, image2, white_image, can_compute_on_white in metric_runs:
                    if metric_name not in self._metric_names:
                        continue

                    def do_it():
                        try:
                            value = metric_fn(image1, image2)
                            if self._normalize_by_white_score and can_compute_on_white:
                                assert white_image is not None
                                value_white: float = metric_fn(image2, white_image)
                                value = (value - value_white) / (1.0 - value_white)
                            return {"value": value}
                        except Exception as e:
                            return {"error": str(e)}

                    response_metric, _ = self._cache.get(
                        {
                            "metric_name": metric_name,
                            "ref_image_path": ref_media_object.location,
                            "generated_image_path": response["image_path"],
                            "normalize_by_white_score": self._normalize_by_white_score,
                        },
                        do_it,
                    )
                    value: float
                    if "error" in response_metric:
                        hlog(f"Error in metric {metric_name}: {response_metric['error']}")
                        value = 0
                    else:
                        value = response_metric["value"]
                    stats_dict[metric_name].add(value)

                stats_dict[self.COMPILE_METRIC].add(1)  # Compiled

        return list(stats_dict.values())

    def lpips_similarity(self, generated_image: Image.Image, reference_image: Image.Image) -> float:
        if self._lpips_metric is None:
            self._lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="vgg").to(self._device)

        preprocessing = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )
        generated_image_tensor = preprocessing(generated_image)
        reference_image_tensor = preprocessing(reference_image)

        # Add batch dimension (B, C, H, W) since torchmetrics expects batches
        img1 = generated_image_tensor.unsqueeze(0).to(self._device)
        img2 = reference_image_tensor.unsqueeze(0).to(self._device)

        # Compute the LPIPS score
        assert self._lpips_metric is not None
        score: float = self._lpips_metric(img1, img2).detach().item()
        return score
