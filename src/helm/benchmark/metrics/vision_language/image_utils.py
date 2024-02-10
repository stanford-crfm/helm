from scipy.stats import wasserstein_distance
import numpy as np

from helm.common.optional_dependencies import handle_module_not_found_error

try:
    import cv2
    from PIL.Image import Image
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, suggestions=["image2structure"])


def preprocess_image(image: Image, norm_exposure: bool = True) -> np.ndarray:
    """Preprocesses an image for use in metrics.
    Returns a grayscale image stored using int in a numpy array.
    Also normalizes the exposure of the image.
    """
    image = image.convert("L")
    np_image = np.array(image)
    assert np_image.dtype == np.uint8
    if norm_exposure:
        np_image = normalize_exposure(np_image)
    return np_image


def get_histogram(img: np.ndarray) -> np.ndarray:
    """
    Get the histogram of an image using numpy for efficiency. For an 8-bit, grayscale image, the
    histogram will be a 256 unit vector in which the nth value indicates
    the percent of the pixels in the image with the given darkness level.
    The histogram's values sum to 1.
    """
    hist, _ = np.histogram(img, bins=256, range=(0, 255))
    hist = hist.astype(float) / img.size  # Normalize the histogram
    return hist


def normalize_exposure(img: np.ndarray) -> np.ndarray:
    """
    Normalize the exposure of an image using numpy for efficiency.
    """
    img = img.astype(int)
    hist, _ = np.histogram(img, bins=256, range=(0, 255))
    hist = hist.astype(float) / img.size  # Normalize histogram

    # Compute the CDF using numpy's cumsum function
    cdf = np.cumsum(hist)
    # Normalize the CDF
    cdf_normalized = np.uint8(255 * cdf / cdf[-1])

    # Use numpy's fancy indexing for normalization of the image
    normalized = cdf_normalized[img]  # type: ignore

    return normalized.astype(int)


def earth_movers_distance(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """
    Measure the Earth Mover's distance between two images

    Args:
        img_a (np.ndarray): the first image
        img_b (np.ndarray): the second image
    Returns:
        float: the Earth Mover's distance between the images
    """
    hist_a = get_histogram(img_a)
    hist_b = get_histogram(img_b)
    return wasserstein_distance(hist_a, hist_b)


def pixel_similarity(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """
    Measure the pixel-level similarity between two images

    Args:
        img_a (np.ndarray): the first image
        img_b (np.ndarray): the second image
    Returns:
        float: the pixel-level similarity between the images
    """
    height, width = img_a.shape
    return 1.0 - np.sum(np.abs(img_a - img_b)) / (height * width * 255)


def sift_similarity(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """
    Use ORB features to measure image similarity between two numpy arrays representing images.

    Args:
        img_a (np.ndarray): the first image
        img_b (np.ndarray): the second image
    Returns:
        float: the ORB similarity between the images
    """
    if len(img_a.shape) < 3 or len(img_b.shape) < 3:
        raise ValueError("Both images must have 3 channels")

    # Initialize the ORB feature detector
    orb = cv2.ORB_create() if hasattr(cv2, "ORB_create") else cv2.ORB()

    # Find the keypoints and descriptors with ORB
    _, desc_a = orb.detectAndCompute(img_a, None)
    _, desc_b = orb.detectAndCompute(img_b, None)

    # Initialize the brute force matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(desc_a, desc_b)

    # Calculate similarity based on the distance of the matches
    similar_regions = [i for i in matches if i.distance < 70]
    if len(matches) == 0:
        return 0
    return len(similar_regions) / len(matches)
