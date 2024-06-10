from typing import List, Tuple
from tqdm import tqdm

import numpy as np
import math

from helm.common.optional_dependencies import handle_module_not_found_error

try:
    import cv2
    from PIL import Image
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, suggestions=["images"])


def to_gray(img: np.ndarray) -> np.ndarray:
    return np.matmul(img, np.array([[0.299], [0.587], [0.114]]))


def get_most_frequent_color(img: np.ndarray) -> Tuple[np.ndarray, float]:
    """Get the most frequent color in the image and its frequency.

    Args:
    img (np.array): Input image array of shape (height, width, channels).

    Returns:
    Tuple[np.array, float]: Most frequent color and its frequency as a percentage of the total number of pixels.
    """
    # Assert to ensure input is a 3D numpy array
    assert len(img.shape) == 3, "Input image must be a 3D numpy array"

    # Reshape image array to 2D (pixel, RGB)
    pixels = img.reshape(-1, img.shape[2])

    # Find unique rows (colors) and their counts
    unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)

    # Find the index of the most frequent color
    most_frequent_color_index = np.argmax(counts)

    # Most frequent color
    most_frequent_color = unique_colors[most_frequent_color_index]

    # Calculate frequency percentage
    frequency = counts[most_frequent_color_index] / pixels.shape[0]

    return most_frequent_color, frequency


def img_to_sig_patches(
    img: np.ndarray,
    rgb_most_frequent_color: np.ndarray,
    patch_size: Tuple[int, int],
    weight_most_frequent_color: float = 0.01,
):
    """
    Convert an RGB image to a signature for cv2.EMD, processing the image in patches.

    Args:
    - img: A 3D numpy array representing an RGB image (height, width, channels).
    - rgb_most_frequent_color: The most frequent color in the image.
    - patch_size: Tuple indicating the height and width of the patches.
    - weight_most_frequent_color: The weight assigned to the most frequent color in the patches.

    Returns:
    - A numpy array suitable for cv2.EMD, containing color values and coordinates of each patch.
        The shape is (num_patches, patch_size[0] * patch_size[1] + 3).
    """
    assert len(img.shape) == 3, "Input image must be a 3D numpy array"

    # Ensure img is a numpy array of type float32
    img = np.array(img, dtype=np.float32)

    # Determine padding needs
    pad_height = (-img.shape[0]) % patch_size[0]
    pad_width = (-img.shape[1]) % patch_size[1]

    # Adjust padding for RGB channels
    padding = ((0, pad_height), (0, pad_width), (0, 0))
    pad_values = (
        (rgb_most_frequent_color[0], rgb_most_frequent_color[0]),
        (rgb_most_frequent_color[1], rgb_most_frequent_color[1]),
        (rgb_most_frequent_color[2], rgb_most_frequent_color[2]),
    )

    # Find the most frequent color for padding
    if pad_height > 0 or pad_width > 0:
        img = np.pad(img, padding, "constant", constant_values=pad_values)
    img /= 255.0  # Normalize colors to [0, 1]

    # Collapse color dimensions to grayscale
    img = to_gray(img)

    # Reshape image into patches and flatten the color dimensions within each patch
    patches = (
        img.reshape(
            (img.shape[0] // patch_size[0], patch_size[0], img.shape[1] // patch_size[1], patch_size[1], img.shape[2])
        )
        .transpose(0, 2, 1, 3, 4)
        .reshape(-1, *patch_size, img.shape[2])
    )

    # Calculate patch positions
    patch_positions = (
        np.mgrid[0 : img.shape[0] // patch_size[0], 0 : img.shape[1] // patch_size[1]].transpose(1, 2, 0).reshape(-1, 2)
    )

    # Normalize positions
    patch_positions = patch_positions / np.array([img.shape[0] // patch_size[0], img.shape[1] // patch_size[1]])

    # Compute the weight of each patch
    # The weight of each point is 1 if the color is not the most frequent color, weight_most_frequent_color otherwise
    flattened_patches = patches.reshape(patches.shape[0], -1)
    gray_most_frequent_color: float = float(to_gray(rgb_most_frequent_color).squeeze() / 255.0)
    weight = weight_most_frequent_color + (1 - weight_most_frequent_color) * np.any(
        flattened_patches != gray_most_frequent_color, axis=1, keepdims=True
    ).astype(np.float32)
    weight /= np.sum(weight)

    # Flatten patches and concatenate with their normalized positions and weights
    sig = np.hstack((weight, flattened_patches, patch_positions))

    return sig.astype(np.float32)


def pad(small_image: Image.Image, large_image: Image.Image, axis: int) -> Image.Image:
    """Pad the axis of the small image to match the size of the large image."""
    new_dim: List[int] = list(small_image.size)
    new_dim[axis] = large_image.size[axis]
    new_dim_tupe: Tuple[int, int] = tuple(new_dim)  # type: ignore
    new_image: Image.Image = Image.new("RGB", new_dim_tupe, (255, 255, 255))
    new_image.paste(small_image, (0, 0))
    return new_image


def reshape_sub_sig_batch(
    sub_sigs: np.ndarray,
    patch_size: Tuple[int, int],
    gray_most_frequent_color: float,
    weight_most_frequent_color: float = 0.01,
) -> np.ndarray:
    """
    Reshape a patch-based signature of an image (Shape: (num_patches, patch_size[0] * patch_size[1] + 3))
    to a batch of signatures for each patch (Shape: (num_patches, patch_size[0] * patch_size[1], 4)).
    Basically goes from a signature on the patch level to a batch of signatures on the pixel level.

    Args:
    - sub_sigs: A numpy array of shape (num_patches, patch_size[0] * patch_size[1] + 1) representing the
        sub-signatures. (the spatial info should have been stripped).
    - patch_size: Tuple indicating the height and width of the patches.
    - gray_most_frequent_color: The most frequent color in the image.
        This is used to reduce the weight assigned to the most frequent color in the patches.
    - weight_most_frequent_color: The weight assigned to the most frequent color in the patches.

    Returns:
    - A numpy array of shape (num_patches, patch_size[0] * patch_size[1], 4) representing the sub-signatures of
        each patch (pixel-level signatures).
    """
    # Ensure sub_sigs has the correct shape
    num_patches = sub_sigs.shape[0]
    flat_patch_size = patch_size[0] * patch_size[1]
    assert sub_sigs.shape[1] == flat_patch_size + 1, f"Expected {flat_patch_size + 1} columns, got {sub_sigs.shape[1]}."

    # Ensure sub_sigs is reshaped to include an extra dimension for concatenation
    num_channels: int = int(round(sub_sigs.shape[0] * sub_sigs.shape[1] / (num_patches * flat_patch_size)))
    assert num_channels == 1, "Only grayscale images are supported for now."
    sub_sigs_reshaped = sub_sigs[:, 1:].reshape(num_patches, flat_patch_size, num_channels)

    # Generate spatial information
    x = np.arange(patch_size[0]) / patch_size[0]
    y = np.arange(patch_size[1]) / patch_size[1]
    x, y = np.meshgrid(x, y)
    spatial_info = np.stack((x.ravel(), y.ravel()), axis=1)  # Shape: (flat_patch_size, 2)

    # Repeat spatial_info for each patch
    spatial_info_repeated = np.repeat(
        spatial_info[np.newaxis, :, :], num_patches, axis=0
    )  # Shape: (num_patches, flat_patch_size, 2)

    # The weight of each point is 1 if the color is not the most frequent color, weight_most_frequent_color otherwise
    # The weight of a pixel is the product of the weight of the patch and the weight of the pixel in the patch
    local_weights = weight_most_frequent_color + (1 - weight_most_frequent_color) * (
        sub_sigs_reshaped != gray_most_frequent_color
    ).astype(np.float32)
    global_weights = sub_sigs[:, 0:1]
    local_weights *= global_weights.reshape(-1, 1, 1)
    local_weights /= np.sum(local_weights, axis=1, keepdims=True)

    # Concatenate sub_sigs with weights and spatial information
    sub_sigs_with_spatial_info = np.concatenate(
        (local_weights, sub_sigs_reshaped, spatial_info_repeated), axis=2
    )  # Shape: (num_patches, flat_patch_size, 4)

    return sub_sigs_with_spatial_info


def compute_cost_matrix_on_sig(
    sig1: np.ndarray,
    sig2: np.ndarray,
    gray_most_frequent_color: float,
    patch_size: Tuple[int, int],
    dim: Tuple[int, int],
    weight_most_frequent_color: float = 0.01,
    use_tqdm: bool = True,
) -> np.ndarray:
    """
    Compute the cost matrix for the EMD between two signatures with pre-reshaping optimization.

    Args:
    - sig1: A numpy array of shape (num_patches, patch_size[0] * patch_size[1] + 2) representing the first signature.
    - sig2: A numpy array of shape (num_patches, patch_size[0] * patch_size[1] + 2) representing the second signature.
    - gray_most_frequent_color: The most frequent color in the images, used to filter out patches that are constant
        equal to the most frequent color.
    - patch_size: Tuple indicating the height and width of the patches.
    - use_tqdm: Boolean indicating whether to display a progress bar.

    Returns:
    - A numpy array of shape (num_patches, num_patches) representing the cost matrix.
    """
    assert sig1.shape == sig2.shape

    # Reshape the sub-signatures at the beginning
    sig1_reshaped = reshape_sub_sig_batch(
        sig1[:, :-2], patch_size, gray_most_frequent_color, weight_most_frequent_color
    ).astype(np.float32)
    sig2_reshaped = reshape_sub_sig_batch(
        sig2[:, :-2], patch_size, gray_most_frequent_color, weight_most_frequent_color
    ).astype(np.float32)

    cost_matrix = np.zeros((sig1.shape[0], sig2.shape[0]))
    multiplier: float = (patch_size[0] * patch_size[1]) ** 0.5 / (dim[0] + dim[1])
    for i in tqdm(range(sig1.shape[0]), disable=not use_tqdm):
        for j in range(sig2.shape[0]):
            pos_sig1 = sig1[i, -2:]
            pos_sig2 = sig2[j, -2:]
            sub_sig1 = sig1_reshaped[i]
            sub_sig2 = sig2_reshaped[j]
            emd_value, _, _ = cv2.EMD(sub_sig1, sub_sig2, cv2.DIST_L1)
            cost_matrix[i, j] = emd_value + np.linalg.norm(pos_sig1 - pos_sig2, 1) * multiplier  # Use L1
    return cost_matrix.astype(np.float32)


def compute_emd_recursive(
    img1_PIL: Image.Image,
    img2_PIL: Image.Image,
    threshold_most_frequent_color: float = 0.5,
    patch_size: Tuple[int, int] = (8, 8),
    max_num_patches: int = 100,
    weight_most_frequent_color: float = 0.001,
    use_tqdm: bool = False,
):
    """
    Compute the Earth Mover's Distance between two images using a recursive approach.
    Both images are discretized into patches, and the EMD is computed on the patches.
    This is done by computing a cost matrix C such that C[i, j] is the cost of moving
    the patch i of img1 to the patch j of img2.

    Moving a patch to another patch has a cost that is not proportional to the number of pixels
    as this corresponds to moving an entire part of the image to another part.

    Args:
    - img1_PIL: A PIL Image representing the first image.
    - img2_PIL: A PIL Image representing the second image (should be the reference if there is one
        as it is used to determine the most frequent color).
    - threshold_most_frequent_color: The threshold under which a color is considered as the most frequent color.
        Constant patches equal to the most frequent color are ignored if the frequency is above this threshold.
    - patch_size: Tuple indicating the height and width of the patches.
    - max_num_patches: The maximum number of patches to use for the EMD computation.
        This is done to avoid having a too long computation time. The images will be resized if necessary.
    - weight_most_frequent_color: The weight assigned to the most frequent color in the patches.
        Should be between 0 and 1 (usually low as the most frequentcolor does not carry much information).
    - use_tqdm: Boolean indicating whether to display a progress bar.

    Returns:
    - A float representing the Earth Mover's Distance between the images.
    """
    assert img1_PIL.size == img2_PIL.size
    assert patch_size[0] > 0 and patch_size[1] > 0
    assert 0 < threshold_most_frequent_color <= 1
    assert max_num_patches > 0
    assert 0 < weight_most_frequent_color <= 1

    # Convert the images to RGB first. Some images have 4 channels (RGBA)
    img1_PIL = img1_PIL.convert("RGB")
    img2_PIL = img2_PIL.convert("RGB")

    # Resize the images so that there are not too many patches
    # Try to maintain the aspect ratio and resize to a multiple of the patch size
    num_patches = math.ceil(img1_PIL.size[0] / patch_size[0]) * math.ceil(img1_PIL.size[1] / patch_size[1])
    if num_patches > max_num_patches:
        ideal_divider = (num_patches / max_num_patches) ** 0.5
        closest_round_width = math.ceil((img1_PIL.size[0] / patch_size[1]) / ideal_divider) * patch_size[1]
        num_patches_width = closest_round_width / patch_size[0]
        # Chooses a round height such that:
        # - (round_width / patch_size[1]) * (round_height / patch_size[0]) <= max_num_patches
        # - the ratio is as unchanged as possible:
        #   (original_width / round_width) / (original_height / round_height) is close to 1
        closest_round_height = math.floor(max_num_patches / num_patches_width) * patch_size[0]
        # Resize the images
        img1_PIL = img1_PIL.resize((closest_round_width, closest_round_height))
        img2_PIL = img2_PIL.resize((closest_round_width, closest_round_height))

    # Convert the images to numpy arrays
    img1_np = np.array(img1_PIL)
    img2_np = np.array(img2_PIL)

    # Get the patch-signature of the images.
    # This is of shape (num_patches, patch_size[0] * patch_size[1] + 3)
    # Each row is a patch, and the columns are:
    # - index 0: weight of the patch
    # - index 1 - 1 + patch_size[0] * patch_size[1]: color values of the patch
    # - index -2, -1: position of the patch
    (rgb_most_frequent_color, frequency) = get_most_frequent_color(img2_np)
    gray_most_frequent_color = float(to_gray(rgb_most_frequent_color).squeeze() / 255.0)
    sig1 = img_to_sig_patches(img1_np, rgb_most_frequent_color, patch_size, weight_most_frequent_color)
    sig2 = img_to_sig_patches(img2_np, rgb_most_frequent_color, patch_size, weight_most_frequent_color)

    if frequency > threshold_most_frequent_color:
        # Ignore patches that are constant equal to the most frequent color
        mask1 = np.any(sig1[:, 1:-2] != gray_most_frequent_color, axis=1)
        mask2 = np.any(sig2[:, 1:-2] != gray_most_frequent_color, axis=1)
        mask = np.logical_or(mask1, mask2)
        sig1 = sig1[mask]
        sig2 = sig2[mask]

    # Normalize the weights
    weight1 = sig1[:, 0]
    weight2 = sig2[:, 0]
    weights = np.maximum(weight1, weight2)
    weights /= np.sum(weights)
    sig1[:, 0] = weights
    sig2[:, 0] = weights

    # Compute EMD
    cost = compute_cost_matrix_on_sig(
        sig1=sig1,
        sig2=sig2,
        gray_most_frequent_color=gray_most_frequent_color,
        patch_size=patch_size,
        dim=img1_PIL.size,
        weight_most_frequent_color=weight_most_frequent_color,
        use_tqdm=use_tqdm,
    )
    emd_value, _, _ = cv2.EMD(sig1, sig2, cv2.DIST_USER, cost)
    return emd_value
