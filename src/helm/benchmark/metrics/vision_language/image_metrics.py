from typing import List, Dict, Optional, Callable, Tuple, Any, Set
from dataclasses import dataclass
from torchvision import transforms, models
from skimage.metrics import structural_similarity as ssim
from nltk.tokenize.treebank import TreebankWordTokenizer

import torch
import warnings
import numpy as np
import os
import tempfile

from helm.benchmark.metrics.copyright_metrics import _edit_similarity
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
    pixel_similarity,
    sift_similarity,
)
from helm.benchmark.metrics.vision_language.emd_utils import compute_emd_recursive, get_most_frequent_color, to_gray

try:
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    from PIL import Image
    import imagehash
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, suggestions=["image2struct"])


def pad(small_image: Image.Image, large_image: Image.Image, axis: int) -> Image.Image:
    """Pad the axis of the small image to match the size of the large image."""
    new_dim: List[int] = list(small_image.size)
    new_dim[axis] = large_image.size[axis]
    new_dim_tupe: Tuple[int, int] = tuple(new_dim)  # type: ignore
    new_image: Image.Image = Image.new("RGB", new_dim_tupe, (255, 255, 255))
    new_image.paste(small_image, (0, 0))
    return new_image


class CompilationError(Exception):
    pass


@dataclass
class AnnotatedMetric:
    name: str
    function: Callable
    input_type: str


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
    EARTH_MOVER_SIMILARITY = "earth_mover_similarity"
    BLOCK_EMD: str = "block_emd"
    PIXEL_SIMILARITY: str = "pixel_similarity"
    SIFT_SIMILARITY: str = "sift_similarity"
    LPIPS_SIMILARITY: str = "lpips_similarity"
    SSIM_SIMILARITY: str = "ssim_similarity"
    FID_SIMILARITY: str = "fid_similarity"
    EDIT_SIMILARITY: str = "edit_similarity"
    NORMALIZE_FID_FACTOR: float = 0.0025

    SIZE_HANDLING_METHODS: List[str] = ["resize", "padding", "none"]

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
        self._tokenizer = TreebankWordTokenizer()

        metrics: List[AnnotatedMetric] = [
            AnnotatedMetric(self.PIXEL_SIMILARITY, pixel_similarity, "image_np_gray"),
            AnnotatedMetric(self.SIFT_SIMILARITY, sift_similarity, "image_np"),
            AnnotatedMetric(self.BLOCK_EMD, self.compute_block_emd_raw, "image_PIL"),  # Raw block-EMD
            AnnotatedMetric(
                self.EARTH_MOVER_SIMILARITY, self.ems, "image_PIL"
            ),  # Normalized block-EMD against black/white
            AnnotatedMetric(self.LPIPS_SIMILARITY, self.lpips_similarity, "image_PIL"),
            AnnotatedMetric(self.FID_SIMILARITY, self.fid_similarity, "image_PIL"),
            AnnotatedMetric(self.SSIM_SIMILARITY, self.compute_ssim, "image_np_gray"),
            AnnotatedMetric(self.EDIT_SIMILARITY, self.compute_edit_sim, "text_str"),
        ]
        self.metrics: Dict[str, AnnotatedMetric] = {metric.name: metric for metric in metrics}

    def _get_compilation_cache_key(self, completion: str) -> Dict[str, str]:
        return {
            "generation_type": self.generation_type,
            "completion": completion,
        }

    def _prepare_inputs(
        self,
        inputs_required: Set[str],
        request_state: RequestState,
        annotation: Dict[str, Any],
        ref_image: Optional[Image.Image],
    ) -> Dict[str, Tuple[Any, Any]]:
        inputs: Dict[str, Tuple[Any, Any]] = {}

        # Image
        if any([input_type.startswith("image") for input_type in inputs_required]):
            # Get the image and make sure we have a reference image
            assert ref_image is not None
            assert "media_object" in annotation
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
                elif self._size_handling_method == "padding":
                    for axis in range(2):
                        if image.size[axis] < ref_image.size[axis]:
                            image = pad(image, ref_image, axis)
                        elif image.size[axis] > ref_image.size[axis]:
                            ref_image = pad(ref_image, image, axis)
                else:
                    raise ValueError(f"size handling method {self._size_handling_method} not recognized.")
            assert image.size == ref_image.size

            # Save the inputs
            inputs["image_PIL"] = (image, ref_image)

            # Convert to numpy array
            if "image_np" in inputs_required:
                rgb_ref_image: np.ndarray = np.array(ref_image)
                rgb_image: np.ndarray = np.array(image)
                inputs["image_np"] = (rgb_image, rgb_ref_image)
            if "image_np_gray" in inputs_required:
                gray_ref_image: np.ndarray = preprocess_image(ref_image)
                gray_image: np.ndarray = preprocess_image(image)
                inputs["image_np_gray"] = (gray_image, gray_ref_image)

        # Text
        if any([input_type.startswith("text") for input_type in inputs_required]):
            assert "text" in annotation
            text: str = annotation["text"]
            reference = request_state.instance.references[0]
            inputs["text_str"] = (text, reference.output.text)

        # Check that all inputs are present
        SUPPORTED_INPUTS: List[str] = ["image_PIL", "image_np", "image_np_gray", "text_str"]
        for input_type in inputs_required:
            if input_type not in SUPPORTED_INPUTS:
                raise AssertionError(f"Input type {input_type} is not supported.")
            if input_type not in inputs:
                raise ValueError(f"Input type {input_type} is required for the metrics but not present.")

        return inputs

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        compiler_name: str = f"{self.generation_type}_compiler"
        if self._cache is None:
            self._cache = metric_service.get_cache(f"image_metrics_{self.generation_type}")

        stats_dict: Dict[str, Stat] = {
            name: Stat(MetricName(name)) for name in (self._metric_names + [self.COMPILE_METRIC])
        }

        if request_state.annotations is None or request_state.result is None:
            raise ValueError(
                "Annotations and results should be present.",
                " Please make sure to add a compiler annotator to the run spec.",
            )
        if compiler_name not in request_state.annotations:
            raise ValueError(f"Compiler {compiler_name} should be present in the annotations.")

        inputs_required: Set[str] = set()
        for metric_name in self._metric_names:
            inputs_required.add(self.metrics[metric_name].input_type)

        # Get the image reference (only once as opening an image is slow)
        # The text annotation can be loaded several times without performance issues
        reference = request_state.instance.references[0]
        ref_image: Optional[Image.Image] = None
        if any([input_type.startswith("image") for input_type in inputs_required]):
            assert reference.output.multimedia_content is not None
            assert len(reference.output.multimedia_content.media_objects) > 0
            ref_media_object: MediaObject = reference.output.multimedia_content.media_objects[0]
            assert ref_media_object.type == "image"
            if ref_media_object.is_local_file and ref_media_object.location is not None:
                ref_image = open_image(ref_media_object.location)
            else:
                raise Exception(
                    "Remote images are not supported in metrics. "
                    "Images should be downloaded when constructing the instance."
                )

        # For each completion, evaluate the metrics
        assert request_state.result is not None
        for completion_index in range(len(request_state.result.completions)):
            annotation: Dict[str, Any] = request_state.annotations[compiler_name][completion_index]

            # Handle errors in annotation
            if "unknown_error" in annotation:
                hlog(
                    f"Unknown error in annotation: {annotation['unknown_error']}\n"
                    f"Scores of zero will be returned for all metrics."
                )
            if "error" in annotation or "unknown_error" in annotation:
                stats_dict[self.COMPILE_METRIC].add(0)  # Did not compile
                # For all other metrics, we set the value to zero
                for metric_name in self._metric_names:
                    stats_dict[metric_name].add(0)
                continue

            # Get te inputs
            inputs = self._prepare_inputs(inputs_required, request_state, annotation, ref_image)

            # Hash the images for the cache key
            hash_dict: Optional[Dict[str, str]] = None
            if "image_PIL" in inputs:
                (image, _) = inputs["image_PIL"]
                hash_dict = {
                    "reference_image": str(AnnotatedImageMetrics.HASH_FUNC(ref_image, hash_size=self.HASH_LENGTH)),
                    "generated_image": str(AnnotatedImageMetrics.HASH_FUNC(image, hash_size=self.HASH_LENGTH)),
                }

            # Evaluate the metrics
            for metric_name in self._metric_names:
                metric: AnnotatedMetric = self.metrics[metric_name]
                (pred, gt) = inputs[metric.input_type]

                value: float
                try:

                    def do_it():
                        value = metric.function(pred, gt)
                        return {"value": value}

                    cache_key = {"metric_name": metric_name, "pred": pred, "gt": gt}
                    if not isinstance(pred, str):
                        assert hash_dict is not None
                        cache_key = {"metric_name": metric_name, **hash_dict}
                    response_metric, _ = self._cache.get(cache_key, do_it)
                    value = response_metric["value"]
                except Exception as e:
                    hlog(f"Error in metric {metric_name}: {str(e)}")
                    value = 0
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
                # https://lightning.ai/docs/torchmetrics/stable/image/learned_perceptual_image_patch_similarity.html
                self._lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).to(
                    self._device
                )

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
        score: float = 1.0 - self._lpips_metric(img1, img2).detach().item()
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

            def load_inception_model():
                return models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, transform_input=False).to(
                    self._device
                )

            try:
                self._inception_model = load_inception_model()
            except PermissionError:
                # If access denied, use a temporary directory
                hlog("Access denied to torch cache directory. Using a temporary directory.")
                temp_cache_dir = tempfile.mkdtemp()
                os.environ["TORCH_HOME"] = temp_cache_dir
                self._inception_model = load_inception_model()
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

        # TODO: Justify the value of the constant here or remove this code to only keep the cosine similarity.
        # fid_score = self._calculate_fid(features1, features2)
        # normalize_fid: float = np.exp(-fid_score * self.NORMALIZE_FID_FACTOR)
        # return normalize_fid

        # Use the cosine similarity between the features as a proxy for FID
        # Return a score between 0 and 1, where 1 is the most similar
        score = 0.5 * (1 + np.dot(features1[0], features2[0]) / (np.linalg.norm(features1) * np.linalg.norm(features2)))
        return score

    def compute_ssim(self, generated_image: np.ndarray, reference_image: np.ndarray) -> float:
        """Compute the Structural Similarity Index (SSIM) between the generated and reference images."""
        # Add 1 and divide by 2 to get a normalized score between 0 and 1, where 1 is the most similar
        return (ssim(generated_image, reference_image) + 1) / 2

    def compute_edit_sim(self, completion: str, reference: str) -> float:
        # `reference` is the entire remaining book for each instance.
        # Truncate it here to be of the same length as the completion to ensure edit-distance is meaningful.
        truncated_reference: str = reference[: len(completion)]

        completion_tokens = self._tokenizer.tokenize(completion)
        truncated_reference_tokens = self._tokenizer.tokenize(truncated_reference)

        # Exploit numpy SIMD for efficiency on CPUs.
        completion_tokens = np.array(completion_tokens)
        truncated_reference_tokens = np.array(truncated_reference_tokens)

        result = _edit_similarity(completion_tokens, truncated_reference_tokens)
        return result

    def ems(
        self,
        pred_image: Image.Image,
        ref_image: Image.Image,
        threshold_most_frequent_color: float = 0.5,
        patch_size: Tuple[int, int] = (8, 8),
        max_num_patches: int = 100,
        weight_most_frequent_color: float = 0.001,
        use_tqdm: bool = False,
    ):
        """Same as compute_emd_similarity_recursive EXCEPT that
        the normalization is against an image of the median color.
        """

        def compute_numerator():
            return self.compute_block_emd_raw_wrapper(
                pred_image,
                ref_image,
                threshold_most_frequent_color,
                patch_size,
                max_num_patches,
                weight_most_frequent_color,
                use_tqdm,
            )

        def compute_denominator():
            ref_img_np = np.array(ref_image)
            (rgb_most_frequent_color, _) = get_most_frequent_color(ref_img_np)
            grayscale_most_frequent_color = to_gray(rgb_most_frequent_color)[0]

            # Most frequent color as base
            if grayscale_most_frequent_color < 127:
                constant_image = Image.new("RGB", ref_image.size, (255, 255, 255))  # Make it white
            else:
                constant_image = Image.new("RGB", ref_image.size, (0, 0, 0))  # Make it black
            value = compute_emd_recursive(
                constant_image,
                ref_image,
                threshold_most_frequent_color,
                patch_size,
                max_num_patches,
                weight_most_frequent_color,
                use_tqdm,
            )
            return {"value": value}

        hash_dict = {
            "reference_image": str(AnnotatedImageMetrics.HASH_FUNC(ref_image, hash_size=self.HASH_LENGTH)),
            "generated_image": str(AnnotatedImageMetrics.HASH_FUNC(pred_image, hash_size=self.HASH_LENGTH)),
        }
        cache_key_numerator = {"metric_name": f"intermediate_{self.BLOCK_EMD}", **hash_dict}
        cache_key_denominator = {"metric_name": "intermediate_ems_extreme_denominator", **hash_dict}

        assert self._cache is not None
        emd_raw, _ = self._cache.get(cache_key_numerator, compute_numerator)
        emd_base, _ = self._cache.get(cache_key_denominator, compute_denominator)

        return 1.0 - emd_raw["value"] / emd_base["value"]

    def compute_block_emd_raw(
        self,
        pred_image: Image.Image,
        ref_image: Image.Image,
        threshold_most_frequent_color: float = 0.5,
        patch_size: Tuple[int, int] = (8, 8),
        max_num_patches: int = 100,
        weight_most_frequent_color: float = 0.001,
        use_tqdm: bool = False,
    ):
        def compute():
            return self.compute_block_emd_raw_wrapper(
                pred_image,
                ref_image,
                threshold_most_frequent_color,
                patch_size,
                max_num_patches,
                weight_most_frequent_color,
                use_tqdm,
            )

        hash_dict = {
            "reference_image": str(AnnotatedImageMetrics.HASH_FUNC(ref_image, hash_size=self.HASH_LENGTH)),
            "generated_image": str(AnnotatedImageMetrics.HASH_FUNC(pred_image, hash_size=self.HASH_LENGTH)),
        }
        cache_key = {"metric_name": f"intermediate_{self.BLOCK_EMD}", **hash_dict}
        assert self._cache is not None
        emd_raw, _ = self._cache.get(cache_key, compute)

        return emd_raw["value"]

    def compute_block_emd_raw_wrapper(
        self,
        pred_image: Image.Image,
        ref_image: Image.Image,
        threshold_most_frequent_color: float = 0.5,
        patch_size: Tuple[int, int] = (8, 8),
        max_num_patches: int = 100,
        weight_most_frequent_color: float = 0.001,
        use_tqdm: bool = False,
    ):
        """Computes the block Earth Moving Distance (EMD). This attempts to
        speed up EMD for images with huge areas by considering
        movement/transformation of blocks of pixels.
        """
        emd_value = compute_emd_recursive(
            pred_image,
            ref_image,
            threshold_most_frequent_color,
            patch_size,
            max_num_patches,
            weight_most_frequent_color,
            use_tqdm,
        )
        return {"value": emd_value}
