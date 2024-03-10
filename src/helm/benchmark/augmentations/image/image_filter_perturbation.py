import copy
import os
import numpy as np
from abc import ABC, abstractmethod
from random import Random

from helm.benchmark.augmentations.perturbation_description import PerturbationDescription
from helm.benchmark.augmentations.perturbation import ImagePerturbation
from helm.common.general import ensure_directory_exists, handle_module_not_found_error


try:
    import cv2
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, suggestions=["heim"])


class ImageFilterPerturbation(ImagePerturbation, ABC):
    name: str = "image_filter"

    # TODO: hardcoded for now. Make this configurable
    cache_path: str = "prod_env/cache/image_perturbations"

    @property
    def description(self) -> PerturbationDescription:
        return PerturbationDescription(name=self.name, robustness=True)

    def perturb(self, image_path: str, rng: Random) -> str:
        """How to perturb the image at `image_path`. Generates a new image and returns the path to it."""
        print(image_path)
        new_image_path: str = self._generate_new_path(image_path)
        if not os.path.exists(new_image_path):
            image = copy.deepcopy(cv2.imread(image_path))
            perturbed_image = self.apply_cv_filter(image)
            cv2.imwrite(new_image_path, perturbed_image)

        return new_image_path

    def _generate_new_path(self, image_path: str) -> str:
        ensure_directory_exists(self.cache_path)
        image_file_name: str = image_path.split("/")[-1]
        new_image_path: str = f"{self.cache_path}/{self.name}_{image_file_name}"
        return new_image_path

    @abstractmethod
    def apply_cv_filter(self, image: np.ndarray) -> np.ndarray:
        """Applies a filter to the image"""
        raise NotImplementedError


class BlurredImageFilterPerturbation(ImageFilterPerturbation):
    """
    Instagram-like filters: Blurred
    Source: https://towardsdatascience.com/python-opencv-building-instagram-like-image-filters-5c482c1c5079
    """

    name: str = "blurred_image_filter"

    def apply_cv_filter(self, image: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(image, (21, 21), 0)


class SepiaImageFilterPerturbation(ImageFilterPerturbation):
    """
    Instagram-like filters: Sepia
    Source: https://towardsdatascience.com/python-opencv-building-instagram-like-image-filters-5c482c1c5079
    """

    name: str = "sepia_image_filter"

    def apply_cv_filter(self, image: np.ndarray) -> np.ndarray:
        kernel = np.array([[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]])
        return cv2.filter2D(image, -1, kernel)


class SharpenedImageFilterPerturbation(ImageFilterPerturbation):
    """
    Instagram-like filters: Sharpened
    Source: https://towardsdatascience.com/python-opencv-building-instagram-like-image-filters-5c482c1c5079
    """

    name: str = "sharpened_image_filter"

    def apply_cv_filter(self, image: np.ndarray) -> np.ndarray:
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        return cv2.filter2D(image, -1, kernel)
