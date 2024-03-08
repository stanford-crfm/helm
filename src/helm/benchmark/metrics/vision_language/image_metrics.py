from typing import List, Dict, Optional, Callable
from torchvision import transforms, models
from skimage.metrics import structural_similarity as ssim
from nltk.tokenize.treebank import TreebankWordTokenizer

import torch
import warnings
import numpy as np

from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.common.images_utils import open_image
from helm.common.gpu_utils import get_torch_device
from helm.common.cache import Cache
from helm.benchmark.adaptation.request_state import RequestState
from helm.common.media_object import MediaObject
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.common.hierarchical_logger import hlog
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.statistic import Stat
from helm.benchmark.metrics.vision_language.image_utils import (
    preprocess_image,
    earth_mover_similarity,
    pixel_similarity,
    sift_similarity,
)

try:
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    from PIL import Image
    import imagehash
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, suggestions=["image2structure"])


class CompilationError(Exception):
    pass


class AnnotatedImageMetrics(Metric):
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
    SSIM_SIMILARITY: str = "ssim_similarity"
    FID_SIMILARITY: str = "fid_similarity"
    EDIT_SIMILARITY: str = "edit_similarity"
    NORMALIZE_FID_FACTOR: float = 0.0025

    SIZE_HANDLING_METHODS: List[str] = ["resize", "none"]

    # Hashing (for caching)
    HASH_LENGTH: int = 16
    HASH_FUNC: Callable = imagehash.average_hash

    def __init__(self, generation_type: str, metric_names: List[str], size_handling_method: str = "resize"):
        self.generation_type = generation_type
        self._metric_names: List[str] = metric_names
        self._lpips_metric: Optional[LearnedPerceptualImagePatchSimilarity] = None
        self._inception_model: Optional[models.Inception3] = None
        self._device = get_torch_device()
        self._cache: Optional[Cache] = None
        self._size_handling_method: str = size_handling_method
        self.tokenizer = TreebankWordTokenizer()

    def _get_compilation_cache_key(self, completion: str) -> Dict[str, str]:
        return {
            "generation_type": self.generation_type,
            "completion": completion,
        }

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        if self._cache is None:
            self._cache = metric_service.get_cache(f"image_metrics_{self.generation_type}")

        stats_dict: Dict[str, Stat] = {
            name: Stat(MetricName(name)) for name in (self._metric_names + [self.COMPILE_METRIC])
        }

        if (
            request_state.annotations is None
            or request_state.result is None
            or len(request_state.annotations) != len(request_state.result.completions)
        ):
            raise ValueError(
                "Annotations and results should be present and have the same length.",
                " Please make sure to add a compiler annotator to the run spec.",
            )

        # Get the references (normally a text and an image)
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

        assert request_state.result is not None
        assert len(request_state.annotations) == len(request_state.result.completions)
        for annotation in request_state.annotations:
            if "error" in annotation:
                stats_dict[self.COMPILE_METRIC].add(0)  # Did not compile
                # For all other metrics, we set the value to zero
                for metric_name in self._metric_names:
                    stats_dict[metric_name].add(0)
                continue

            assert "media_object" in annotation, "No media object in the annotation"
            assert isinstance(annotation["media_object"], MediaObject)
            media_object: MediaObject = annotation["media_object"]
            assert media_object.type == "image"
            assert media_object.is_local_file and media_object.location is not None
            image: Image.Image = Image.open(media_object.location).convert("RGB")

            # Handle difference in size
            if image.size != ref_image.size:
                if self._size_handling_method == "none":
                    raise ValueError(
                        "Compiled image and reference image should have the same size"
                        " when the size handling method is none."
                    )
                elif self._size_handling_method == "resize":
                    image = image.resize(ref_image.size)
                else:
                    raise ValueError(f"size handling method {self._size_handling_method} not recognized.")
            assert image.size == ref_image.size

            rgb_image: np.ndarray = np.array(image)
            gray_image: np.ndarray = preprocess_image(image)

            # List of metrics and arguments to evaluate
            # The arguments are as follows:
            # 1. Name of the metric
            # 2. Function to compute the metric
            # 3. The generated image obhect (type can depend on the metric)
            # 4. The reference image object (type can depend on the metric)
            metric_runs: list = [
                # Image
                [self.PIXEL_SIMILARITY, pixel_similarity, gray_image, gray_ref_image],
                [self.SIFT_SIMILARITY, sift_similarity, rgb_image, rgb_ref_image],
                [self.EARTH_MOVER_SIMILARITY, earth_mover_similarity, gray_image, gray_ref_image],
                [self.LPIPS_SIMILARITY, self.lpips_similarity, image, ref_image],
                [self.FID_SIMILARITY, self.fid_similarity, image, ref_image],
                [self.SSIM_SIMILARITY, self.compute_ssim, gray_image, gray_ref_image],
            ]

            if "text" in annotation:
                # For some tasks, we have the reference text. For those, compute the edit similarity for some signal.
                # Although having the reference text is not a requirement for a task to be included in Image2Structure.
                text: str = annotation["text"]
                ref_text: str = reference.output.text
                metric_runs.append(
                    [self.EDIT_SIMILARITY, self.compute_edit_sim, text, ref_text],
                )

            hash_dict = {
                "reference_image": str(AnnotatedImageMetrics.HASH_FUNC(ref_image, hash_size=self.HASH_LENGTH)),
                "generated_image": str(AnnotatedImageMetrics.HASH_FUNC(image, hash_size=self.HASH_LENGTH)),
            }
            for metric_name, metric_fn, pred, gt in metric_runs:
                if metric_name not in self._metric_names:
                    continue

                def do_it():
                    try:
                        value = metric_fn(pred, gt)
                        return {"value": value}
                    except Exception as e:
                        return {"error": str(e)}

                cache_key = {"metric_name": metric_name, "pred": pred, "gt": gt}
                if not isinstance(pred, str):
                    cache_key = {"metric_name": metric_name, **hash_dict}
                response_metric, _ = self._cache.get(cache_key, do_it)
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
        """Compute the LPIPS similarity between the generated and reference images.

        This metric is defined here as it requires loading the LPIPS model.
        Storing the model in this class is easier than passing it as an argument.
        """
        if self._lpips_metric is None:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
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

    def _calculate_fid(self, act1, act2):
        # Directly use the provided activations, assuming they are already means
        mu1, mu2 = act1[0], act2[0]  # Assuming act1 and act2 are of shape (1, 1000)

        # Since we cannot compute a meaningful covariance matrix for single observations,
        # and the provided sigma is scalar (not meaningful in this context),
        # we'll skip the covariance part of the standard FID calculation.
        # This is a significant deviation from the FID's intended use.

        # Compute the square difference between the means
        ssdiff = np.sum((mu1 - mu2) ** 2.0)

        # Placeholder for FID score since we're not using covariance matrices
        fid = ssdiff  # This is not a standard FID calculation.

        return fid

    def _get_inception_features(self, img_tensor):
        if self._inception_model is None:
            self._inception_model = models.inception_v3(
                weights=models.Inception_V3_Weights.IMAGENET1K_V1, transform_input=False
            ).to(self._device)
            self._inception_model.eval()
        with torch.no_grad():
            if self._inception_model.training:
                self._inception_model.eval()
            pred = self._inception_model(img_tensor)
        return pred.cpu().detach().numpy()

    def _preprocess_image(self, image):
        # Source: https://pytorch.org/hub/pytorch_vision_inception_v3/
        preprocess = transforms.Compose(
            [
                transforms.Resize(299),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return preprocess(image)

    def fid_similarity(self, generated_image: Image.Image, reference_image: Image.Image) -> float:
        """Compute the Frechet Inception Distance (FID) between the generated and reference images.

        This metric is defined here as it requires loading the Inception model.
        Storing the model in this class is easier than passing it as an argument.
        """
        img1_tensor = self._preprocess_image(generated_image).unsqueeze(0).to(self._device)
        img2_tensor = self._preprocess_image(reference_image).unsqueeze(0).to(self._device)

        features1 = self._get_inception_features(img1_tensor)
        features2 = self._get_inception_features(img2_tensor)

        fid_score = self._calculate_fid(features1, features2)
        normalize_fid: float = np.exp(-fid_score * self.NORMALIZE_FID_FACTOR)
        return normalize_fid

    def compute_ssim(self, generated_image: np.ndarray, reference_image: np.ndarray) -> float:
        """Compute the Structural Similarity Index (SSIM) between the generated and reference images."""
        return ssim(generated_image, reference_image)

    def compute_edit_sim(self, completion: str, reference: str) -> float:
        from helm.benchmark.metrics.copyright_metrics import _edit_similarity

        # `reference` is the entire remaining book for each instance.
        # Truncate it here to be of the same length as the completion to ensure edit-distance is meaningful.
        truncated_reference: str = reference[: len(completion)]

        completion_tokens = self.tokenizer.tokenize(completion)
        truncated_reference_tokens = self.tokenizer.tokenize(truncated_reference)

        # Exploit numpy SIMD for efficiency on CPUs.
        completion_tokens = np.array(completion_tokens)
        truncated_reference_tokens = np.array(truncated_reference_tokens)

        result = _edit_similarity(completion_tokens, truncated_reference_tokens)
        return result
