import numpy as np

from helm.common.optional_dependencies import handle_module_not_found_error

try:
    import cv2
    from PIL.Image import Image
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, suggestions=["image2struct"])


def preprocess_image(image: Image) -> np.ndarray:
    """Preprocesses an image for use in metrics.
    Returns a grayscale image stored using int in a numpy array.
    Also normalizes the exposure of the image.
    """
    image = image.convert("L")
    np_image = np.array(image)
    assert np_image.dtype == np.uint8
    return np_image


def pixel_similarity(img_a: np.ndarray, img_b: np.ndarray, threshold: float = 0.5, tolerance: float = 0.02) -> float:
    """
    Measure the pixel-level similarity between two images
    If the image has a color that occurs more than 100 * threshold percent of the time,
    Then the associated pixels are ignored and the match is computed only on the other pixels.
    A tolerance is used to compare each pixels to allow some small variations in color.
    The tolerance is between 0 (exact match) and 1 (every color is ok)

    Args:
        img_a (np.ndarray): the first image
        img_b (np.ndarray): the second image
        threshold (float): Threshold to ignore dominant colors.
        tolerance (float): Tolerance for color variation.
    Returns:
        float: the pixel-level similarity between the images (between 0 and 1)
    """
    if img_a.shape != img_b.shape:
        raise ValueError(
            f"Images must have the same dimensions. img_a.shape = {img_a.shape}, img_b.shape = {img_b.shape}"
        )

    # Flatten the images
    img_a_flat = img_a.reshape(-1, img_a.shape[-1])
    img_b_flat = img_b.reshape(-1, img_b.shape[-1])

    # Calculate color differences with tolerance
    color_diff = np.linalg.norm(img_a_flat - img_b_flat, axis=1) / 255
    within_tolerance = color_diff <= tolerance

    # Calculate frequencies of all colors
    unique_colors, indices = np.unique(np.concatenate((img_a_flat, img_b_flat), axis=0), axis=0, return_inverse=True)
    color_counts = np.bincount(indices)

    # Identify colors to ignore based on frequency threshold
    ignore_colors_mask = color_counts > (len(img_a_flat) + len(img_b_flat)) * threshold / 2
    ignore_in_a = ignore_colors_mask[indices[: len(img_a_flat)]]
    ignore_in_b = ignore_colors_mask[indices[len(img_a_flat) :]]

    # Apply ignore mask
    valid_pixels = np.logical_not(np.logical_or(ignore_in_a, ignore_in_b)) & within_tolerance

    # Calculate similarity
    similarity = np.mean(valid_pixels) if len(valid_pixels) > 0 else 0

    return similarity


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
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # Match descriptors.
    matches = bf.match(desc_a, desc_b)

    # Calculate similarity based on the distance of the matches
    similar_regions = [i for i in matches if i.distance < 70]
    if len(matches) == 0:
        return 0
    return len(similar_regions) / len(matches)
