from typing import List, Tuple

from PIL import Image
import cv2
import numpy as np
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
from toolbox.printing import debug
from tqdm import tqdm

import numpy as np
import math
from scipy.stats import mode


def img_to_sig(img):
    """
    Convert an RGB image to a signature for cv2.EMD.

    Parameters:
    - img: A 3D numpy array representing an RGB image (height, width, channels).

    Returns:
    - A numpy array of shape (height*width, 5), suitable for cv2.EMD, containing the
      color values and coordinates of each pixel.
    """
    # Ensure img is a numpy array of type float32
    img = np.array(img, dtype=np.float32)
    num_color_channels = img.shape[2]

    # Generate coordinate grids for the entire array
    x, y = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))

    # Flatten the arrays
    x_flat = x.ravel() / img.shape[1]
    y_flat = y.ravel() / img.shape[0]
    colors_flat = img.reshape(-1, num_color_channels) / 255.0  # Flatten color channels, maintaining the RGB structure

    # Combine color values with their coordinates
    sig = np.hstack((np.ones_like(colors_flat), colors_flat, y_flat[:, np.newaxis], x_flat[:, np.newaxis]))

    return sig.astype(np.float32)


def get_most_frequent_color(img: np.array) -> Tuple[np.array, float]:
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
    debug(frequency)
    debug(most_frequent_color)

    return most_frequent_color, frequency


def compute_cost_matrix(img1, img2, patch_size=(8, 8), to_gray=False):
    """
    Discretize the images into patches, for each couple of patches, call img_to_sig and then compute the EMD
    """
    assert img1.shape == img2.shape
    if to_gray:
        img1 = np.mean(img1, axis=2, keepdims=True)
        img2 = np.mean(img2, axis=2, keepdims=True)

    # Assume no padding is needed
    patches_dim_0 = img1.shape[0] // patch_size[0]
    patches_dim_1 = img1.shape[1] // patch_size[1]

    # Reshape the image to a 5D array of patches: (num_patches_0, num_patches_1, patch_height, patch_width, num_channels)
    reshaped1 = img1[: patches_dim_0 * patch_size[0], : patches_dim_1 * patch_size[1]].reshape(
        patches_dim_0, patch_size[0], patches_dim_1, patch_size[1], img1.shape[2]
    )
    reshaped2 = img2[: patches_dim_0 * patch_size[0], : patches_dim_1 * patch_size[1]].reshape(
        patches_dim_0, patch_size[0], patches_dim_1, patch_size[1], img2.shape[2]
    )
    # The cost matrix is a 2D array of shape (num_patches_0 * num_patches_1, num_patches_0 * num_patches_1)
    cost_matrix = np.zeros((patches_dim_0 * patches_dim_1, patches_dim_0 * patches_dim_1))

    print("Computing cost matrix of size", cost_matrix.shape)
    for i in tqdm(range(patches_dim_0)):
        for j in tqdm(range(patches_dim_1)):
            for ii in range(patches_dim_0):
                for jj in range(patches_dim_1):
                    patch1 = reshaped1[i, j, :, :, :]
                    patch2 = reshaped2[ii, jj, :, :, :]
                    sig1 = img_to_sig(patch1)
                    sig2 = img_to_sig(patch2)
                    cost_matrix[i * patches_dim_1 + j, ii * patches_dim_1 + jj] = cv2.EMD(sig1, sig2, cv2.DIST_L2)[0]
    return cost_matrix.astype(np.float32)


def img_to_sig_patches(img, most_frequent_color: np.array, patch_size=(2, 2), to_gray=False):
    """
    Convert an RGB image to a signature for cv2.EMD, processing the image in patches.

    Parameters:
    - img: A 3D numpy array representing an RGB image (height, width, channels).
    - patch_size: Tuple indicating the height and width of the patches.
    - to_gray: Boolean indicating whether to collapse the color dimension to grayscale.

    Returns:
    - A numpy array suitable for cv2.EMD, containing color values and coordinates of each patch.
    """
    assert len(img.shape) == 3
    # Ensure img is a numpy array of type float32
    img = np.array(img, dtype=np.float32)

    # Determine padding needs
    pad_height = (-img.shape[0]) % patch_size[0]
    pad_width = (-img.shape[1]) % patch_size[1]
    debug(patch_size)
    debug(pad_height)
    debug(pad_width)
    debug(img.shape[0] + pad_height)

    # Adjust padding for RGB channels
    padding = ((0, pad_height), (0, pad_width), (0, 0))
    pad_values = (
        (most_frequent_color[0], most_frequent_color[0]),
        (most_frequent_color[1], most_frequent_color[1]),
        (most_frequent_color[2], most_frequent_color[2]),
    )

    # Find the most frequent color for padding
    if pad_height > 0 or pad_width > 0:
        img = np.pad(img, padding, "constant", constant_values=pad_values)
    img /= 255.0  # Normalize colors to [0, 1]

    # Collapse color dimensions to grayscale if needed
    if to_gray:
        img = np.mean(img, axis=2, keepdims=True)
        # plt.imshow(img[:, :, 0])
        # plt.show()

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

    # Flatten patches and concatenate with their normalized positions
    flattened_patches = patches.reshape(patches.shape[0], -1)
    sig = np.hstack((np.ones_like(flattened_patches), flattened_patches, patch_positions))

    return sig.astype(np.float32)


def pad(small_image: Image.Image, large_image: Image.Image, axis: int) -> Image.Image:
    """Pad the axis of the small image to match the size of the large image."""
    new_dim: List[int] = list(small_image.size)
    new_dim[axis] = large_image.size[axis]
    new_dim_tupe: Tuple[int, int] = tuple(new_dim)  # type: ignore
    new_image: Image.Image = Image.new("RGB", new_dim_tupe, (255, 255, 255))
    new_image.paste(small_image, (0, 0))
    return new_image


def get_translation(image: np.array, translation: Tuple[int, int]) -> np.array:
    """Translate the image by the given translation."""
    copy = np.ones_like(image) * image.max()
    copy[
        max(0, translation[0]) : min(image.shape[0], image.shape[0] + translation[0]),
        max(0, translation[1]) : min(image.shape[1], image.shape[1] + translation[1]),
    ] = image[
        max(0, -translation[0]) : min(image.shape[0], image.shape[0] - translation[0]),
        max(0, -translation[1]) : min(image.shape[1], image.shape[1] - translation[1]),
    ]
    return copy


def reshape_sub_sig_batch(sub_sigs, patch_size, most_frequent_color):
    """
    Reshape a batch of sub-signatures.
    """
    num_patches = sub_sigs.shape[0]
    flat_patch_size = patch_size[0] * patch_size[1]

    # Ensure sub_sigs is reshaped to include an extra dimension for concatenation
    sub_sigs_reshaped = sub_sigs.reshape(num_patches, flat_patch_size, 1)  # Add an extra dimension

    # Generate spatial information
    x = np.arange(patch_size[0]) / patch_size[0]
    y = np.arange(patch_size[1]) / patch_size[1]
    x, y = np.meshgrid(x, y)
    spatial_info = np.stack((x.ravel(), y.ravel()), axis=1)  # Shape: (flat_patch_size, 2)

    # Repeat spatial_info for each patch
    spatial_info_repeated = np.repeat(
        spatial_info[np.newaxis, :, :], num_patches, axis=0
    )  # Shape: (num_patches, flat_patch_size, 2)

    # Concatenate sub_sigs with spatial information
    # The weight of each point is 1 if the color is not the most frequent color, 0 otherwise
    weights = 0.1 + 0.9 * (sub_sigs_reshaped != most_frequent_color).astype(np.float32)
    weights /= np.sum(weights, axis=1, keepdims=True)
    sub_sigs_with_spatial_info = np.concatenate(
        (weights, sub_sigs_reshaped, spatial_info_repeated), axis=2
    )  # Shape: (num_patches, flat_patch_size, 3)

    debug(sub_sigs_with_spatial_info)
    return sub_sigs_with_spatial_info


def compute_cost_matrix_on_sig(sig1, sig2, most_frequent_color, patch_size=(8, 8)):
    """
    Compute the cost matrix for the EMD between two signatures with pre-reshaping optimization.
    """
    assert sig1.shape == sig2.shape

    # Reshape the sub-signatures at the beginning
    debug(sig1)
    debug(sig1[:, 1:-2:2])
    sig1_reshaped = reshape_sub_sig_batch(sig1[:, 1:-2:2], patch_size, most_frequent_color).astype(np.float32)
    sig2_reshaped = reshape_sub_sig_batch(sig2[:, 1:-2:2], patch_size, most_frequent_color).astype(np.float32)

    cost_matrix = np.zeros((sig1.shape[0], sig2.shape[0]))
    for i in tqdm(range(sig1.shape[0])):
        for j in range(sig2.shape[0]):
            pos_sig1 = sig1[i, -2:]
            pos_sig2 = sig2[j, -2:]
            cost_matrix[i, j] = cv2.EMD(sig1_reshaped[i], sig2_reshaped[j], cv2.DIST_L2)[0] + patch_size[
                0
            ] * np.linalg.norm(pos_sig1 - pos_sig2)

    # for i in range(cost_matrix.shape[0]):
    #     debug(np.min(cost_matrix[i]))
    return cost_matrix.astype(np.float32)


def compute_emd_simple(img1: np.array, img2: np.array, to_gray: bool = True, threshold: float = 0.5):
    # Flatten the images to turn them into 1D distributions
    debug(img1)
    debug(img2)
    if to_gray:
        img1 = np.mean(img1, axis=2, keepdims=True)
        img2 = np.mean(img2, axis=2, keepdims=True)
    (most_frequent_color, frequency) = get_most_frequent_color(img1)
    distribution1 = img_to_sig(img1)
    distribution2 = img_to_sig(img2)
    print("Before removing most frequent color")
    debug(distribution1)
    debug(distribution2)
    debug(most_frequent_color)
    debug(frequency)
    if frequency > threshold and to_gray:
        # Ignore patches that are constant equal to the most frequent color
        most_frequent_color = np.mean(most_frequent_color) / 255.0
        # Filter all patches that are equal to the most frequent color on every point except the last 2 that correspond to the coordinates
        # print(distribution1[:10, :-2])
        mask1 = np.any(distribution1[:, 1:2] != most_frequent_color, axis=1)
        mask2 = np.any(distribution2[:, 1:2] != most_frequent_color, axis=1)
        print("Size of mask 1", np.sum(mask1) / mask1.size)
        print("Size of mask 2", np.sum(mask2) / mask2.size)
        mask = np.logical_or(mask1, mask2)
        distribution1 = distribution1[mask]
        distribution2 = distribution2[mask]
    print("After removing most frequent color")
    debug(distribution1)
    debug(distribution2)
    print("")

    # Compute EMD
    # help(cv2.EMD)
    emd_value = cv2.EMD(distribution2, distribution1, cv2.DIST_L1)
    debug(emd_value)
    emd_value = emd_value[0]

    return emd_value


def compute_emd(
    img1: Image.Image,
    img2: Image.Image,
    to_gray: bool = True,
    threshold: float = 0.5,
    patch_size: Tuple[int, int] = (8, 8),
    max_num_patches: int = 100,
):
    # Flatten the images to turn them into 1D distributions
    assert img1.size == img2.size
    debug(img1)
    debug(img2)
    num_patches = math.ceil(img1.size[0] / patch_size[0]) * math.ceil(img1.size[1] / patch_size[1])
    if num_patches > max_num_patches:
        ideal_divider = (num_patches / max_num_patches) ** 0.5
        debug(ideal_divider)
        closest_round_width = math.ceil((img1.size[0] / patch_size[1]) / ideal_divider) * patch_size[1]
        num_patches_width = closest_round_width / patch_size[0]
        debug(closest_round_width)
        debug(num_patches_width)
        # Chooses a round height such that:
        # - (round_width / patch_size[1]) * (round_height / patch_size[0]) <= max_num_patches
        # - the ratio is as unchanged as possible: (original_width / round_width) / (original_height / round_height) is close to 1
        closest_round_height = math.floor(max_num_patches / num_patches_width) * patch_size[0]
        debug(closest_round_height)
        debug(closest_round_height / patch_size[1])

        img1 = img1.resize((closest_round_width, closest_round_height))
        img2 = img2.resize((closest_round_width, closest_round_height))

    img1 = np.array(img1)
    img2 = np.array(img2)
    plt.imshow(img2)
    plt.show()

    debug(num_patches)
    (most_frequent_color, frequency) = get_most_frequent_color(img1)
    distribution1 = img_to_sig_patches(img1, most_frequent_color, patch_size=patch_size, to_gray=to_gray)
    distribution2 = img_to_sig_patches(img2, most_frequent_color, patch_size=patch_size, to_gray=to_gray)
    print("Before removing most frequent color")
    debug(distribution1)
    debug(distribution2)
    debug(most_frequent_color)
    debug(frequency)
    if frequency > threshold and to_gray:
        # Ignore patches that are constant equal to the most frequent color
        most_frequent_color = np.mean(most_frequent_color) / 255.0
        # Filter all patches that are equal to the most frequent color on every point except the last 2 that correspond to the coordinates
        # print(distribution1[:10, :-2])
        mask1 = np.any(distribution1[:, :-2] != most_frequent_color, axis=1)
        mask2 = np.any(distribution2[:, :-2] != most_frequent_color, axis=1)
        print("Size of mask 1", np.sum(mask1) / mask1.size)
        print("Size of mask 2", np.sum(mask2) / mask2.size)
        mask = np.logical_or(mask1, mask2)
        distribution1 = distribution1[mask]
        distribution2 = distribution2[mask]
    print("After removing most frequent color")
    debug(distribution1)
    debug(distribution2)
    print("")

    cost = compute_cost_matrix_on_sig(
        distribution1, distribution2, patch_size=patch_size, most_frequent_color=most_frequent_color
    )
    # plt.imshow(cost)
    # plt.show()

    # Compute EMD
    emd_value, _, flow = cv2.EMD(distribution1, distribution2, cv2.DIST_USER, cost)
    plot_flow(
        distribution1, distribution2, flow=flow, dim=(closest_round_height, closest_round_width), patch_size=patch_size
    )
    debug(emd_value)

    return emd_value


def filter_pixels_by_threshold(img, threshold):
    """
    Identify pixels that exceed the threshold frequency in the image
    and return a mask for all pixels to be kept.
    """
    unique, counts = np.unique(img, return_counts=True)
    frequencies = counts / img.size
    # Pixels to keep are those whose frequencies are below the threshold
    pixels_to_keep = unique[frequencies < threshold]
    mask = np.isin(img, pixels_to_keep)

    # Print the colors that were ignored
    ignored_colors = unique[frequencies >= threshold]
    # if len(ignored_colors) > 0:
    #     print(f"Ignored colors: {ignored_colors}")
    return mask


def compute_emd_with_threshold(img1: np.array, img2: np.array, threshold: float = 0.5):
    """
    Compute the EMD between two images, ignoring pixel values in both images
    that exceed a specified presence threshold in the first image.
    """
    # Identify pixels in img1 that are below the threshold
    mask1 = filter_pixels_by_threshold(img1, threshold)
    # print("Size of mask", np.sum(mask1) / mask1.size)
    # print(np.mean(img1[mask1]))

    # Apply the same for img2 for consistency in comparison
    mask2 = filter_pixels_by_threshold(img2, threshold)
    # print("Size of mask", np.sum(mask2) / mask2.size)
    # print(np.mean(img2[mask2]))

    # Combine masks to focus on common set of pixels for comparison
    combined_mask = np.logical_or(mask1, mask2)

    # Apply the mask and flatten the filtered images to 1D distributions
    filtered_distribution1 = img_to_sig(img1[combined_mask])
    filtered_distribution2 = img_to_sig(img2[combined_mask])
    debug(filtered_distribution1)

    # Compute EMD on filtered distributions
    emd_value = cv2.EMD(filtered_distribution1, filtered_distribution2, cv2.DIST_L2)[0]
    debug(emd_value)

    return emd_value / 255.0


def plot_flow(sig1, sig2, flow, dim, patch_size, arrow_width_scale=3):
    """Plots the flow computed by cv2.EMD

    The source images are retrieved from the signatures and
    plotted in a combined image, with the first image in the
    red channel and the second in the green. Arrows are
    overplotted to show moved earth, with arrow thickness
    indicating the amount of moved earth."""

    img1 = patch_sig_to_img(sig1, dim, patch_size)
    img2 = patch_sig_to_img(sig2, dim, patch_size)
    plt.imshow(img2)
    plt.show()
    combined = np.dstack((img1, img2, 0 * img2))
    # RGB values should be between 0 and 1
    combined /= combined.max()
    print('Red channel is "before"; green channel is "after"; yellow means "unchanged"')
    plt.imshow(combined)

    flows = np.transpose(np.nonzero(flow))
    for src, dest in flows:
        # Skip the pixel value in the first element, grab the
        # coordinates. It'll be useful later to transpose x/y.
        start = sig1[src, 2:][::-1]
        end = sig2[dest, 2:][::-1]
        if np.all(start == end):
            # Unmoved earth shows up as a "flow" from a pixel
            # to that same exact pixel---don't plot mini arrows
            # for those pixels
            continue

        # Add a random shift to arrow positions to reduce overlap.
        shift = np.random.random(1) * 0.3 - 0.15
        start = start + shift
        end = end + shift

        mag = flow[src, dest] * arrow_width_scale
        plt.quiver(
            *start,
            *(end - start),
            angles="xy",
            scale_units="xy",
            scale=1,
            color="white",
            edgecolor="black",
            linewidth=mag / 3,
            width=mag,
            units="dots",
            headlength=5,
            headwidth=3,
            headaxislength=4.5,
        )

    plt.title("Earth moved from img1 to img2")


def sig_to_img(sig, dim):
    """Convert a signature back to a 2D image"""
    img = np.zeros(dim, dtype=float)
    for i in range(sig.shape[0]):
        x = int(sig[i, 3] * dim[1])
        y = int(sig[i, 2] * dim[0])
        img[y, x] = sig[i, 1]
    debug(img)
    return img


def patch_sig_to_img(sig, dim, patch_size):
    """Convert a signature back to a 2D image"""
    debug(sig)
    patches_sig = reshape_sub_sig_batch(sig[:, 1:-2:2], patch_size, np.array([1.0]))
    debug(patches_sig)
    debug(dim)
    img = np.zeros(dim, dtype=float)
    for i in range(patches_sig.shape[0]):
        xx = round(dim[1] * sig[i, -1])
        yy = round(dim[0] * sig[i, -2])
        for j in range(patches_sig.shape[1]):
            y = round(patches_sig[i, j, 3] * patch_size[1])
            x = round(patches_sig[i, j, 2] * patch_size[0])
            img[y + yy, x + xx] = patches_sig[i, j, 1]
    debug(img)
    return img


def display_patch_sizes_from_signature(sig):
    sig_dim = sig.shape[1]
    patch_dim = int((sig_dim - 2) ** 0.5)
    assert patch_dim**2 + 2 == sig_dim
    patch_size = (patch_dim, patch_dim)
    num_patches = sig.shape[0]
    patches = sig[:, :-2].reshape(num_patches, *patch_size)

    # Display the patch sizes on sqrt(num_patches) x sqrt(num_patches) grid
    num_rows = int(num_patches**0.5)
    num_cols = math.ceil(num_patches / num_rows)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))
    for i in range(num_rows):
        for j in range(num_cols):
            if i * num_cols + j < num_patches:
                axes[i, j].imshow(patches[i * num_cols + j])
            axes[i, j].axis("off")
    plt.show()


def image_to_patches(img, patch_size=(32, 32), aggregator=np.mean):
    """
    Convert an RGB image into a list of patches and apply an aggregator function to each patch.

    Parameters:
    - img: A NumPy array representing the RGB image.
    - patch_size: Tuple indicating the height and width of the patches.
    - aggregator: Function to apply to each patch across the spatial dimensions, preserving the channel dimension.

    Returns:
    - A NumPy array of aggregated patch values with shape (num_patches, num_channels), where num_patches is the total number of patches, and num_channels is 3 for RGB images.
    """
    # Calculate the number of patches along each dimension
    patches_dim_0 = img.shape[0] // patch_size[0]
    patches_dim_1 = img.shape[1] // patch_size[1]

    # Reshape the image to a 5D array of patches: (num_patches_0, num_patches_1, patch_height, patch_width, num_channels)
    reshaped = img[: patches_dim_0 * patch_size[0], : patches_dim_1 * patch_size[1]].reshape(
        patches_dim_0, patch_size[0], patches_dim_1, patch_size[1], img.shape[2]
    )

    # Swap axes to group all patch pixels together
    swapped = reshaped.swapaxes(1, 2)

    # Apply the aggregator function to each patch across the spatial dimensions (height and width), preserving the channel dimension
    aggregated = aggregator(swapped, axis=(2, 3))

    # Flatten the array to shape (num_patches, num_channels)
    return aggregated.reshape(-1, img.shape[2])


if __name__ == "__main__":
    # size = (8, 8, 3)
    # white = np.ones(size) * 255
    # black = np.ones(size) * 127
    # black2 = black.copy()
    # # black2[7, 7, :] = 255
    # black2[0, 0, :] = 1
    # black[0, 0, :] = 255
    # debug(compute_emd_simple(black2, black))
    # raise ValueError("done")

    # print(img_to_sig(np.array([[0.5, 0.1], [0.7, 0.2]])))
    num: int = 0
    img_path_ref = f"../images_test/ref_image_{num}.png"
    img_path_pred = f"../images_test/pred_image_{num}.png"
    img_ref = Image.open(img_path_ref)
    img_pred = Image.open(img_path_pred)

    for axis in range(2):
        if img_pred.size[axis] < img_ref.size[axis]:
            img_pred = pad(img_pred, img_ref, axis)
        elif img_pred.size[axis] > img_ref.size[axis]:
            img_ref = pad(img_ref, img_pred, axis)

    # img_ref = img_ref.resize((256, 256))
    # img_pred = img_pred.resize((256, 256))
    # Show the images
    # plt.imshow(img_ref)
    # plt.show()
    # get_most_frequent_color(np.array(img_ref))
    # patch_size = (128, 128)
    # cost_matrix = compute_cost_matrix(np.array(img_ref), np.array(img_pred), patch_size=patch_size, to_gray=True)
    # print(cost_matrix)
    # sig1 = img_to_sig_patches(np.array(img_ref), patch_size=patch_size, to_gray=True)
    # sig2 = img_to_sig_patches(np.array(img_pred), patch_size=patch_size, to_gray=True)
    # print(cv2.EMD(sig1, sig2, cv2.DIST_USER, cost_matrix))

    image_ref_np = np.array(img_ref)
    image_pred_np = np.array(img_pred)
    white_PIL = Image.new("RGB", img_ref.size, (255, 255, 255))

    image_ref_np_patches = image_to_patches(image_ref_np)
    image_pred_np_patches = image_to_patches(image_pred_np)

    for emd_function in [compute_emd]:  # , compute_emd_with_threshold]:
        print("")
        print(f"{emd_function} (white, ref): {emd_function(white_PIL, img_ref)}\n\n\n")
        print(f"{emd_function} (pred, ref): {emd_function(img_pred, img_ref)}")
        # print(
        #     f"{emd_function} (ref - ref): {emd_function(image_ref_np, image_ref_np)} / patches: {emd_function(image_ref_np_patches.copy(), image_ref_np_patches.copy())}"
        # )
        # for translation in [(0, 0), (10, 10), (20, 20), (30, 30), (40, 40), (50, 50)]:
        #     image_pred_translated = get_translation(image_pred_np, translation)
        #     image_pred_translated_patches = image_to_patches(image_pred_translated)
        #     print(
        #         f"{emd_function} (pred - pred_translated_{translation}): {emd_function(image_pred_np.copy(), image_pred_translated)} / patches: {emd_function(image_pred_np_patches.copy(), image_pred_translated_patches)}"
        #     )
        # print(
        #     f"{emd_function}: {emd_function(image_ref_np.copy(), image_pred_np.copy())} / patches: {emd_function(image_ref_np_patches.copy(), image_pred_np_patches.copy())}"
        # )
        # print(
        #     f"{emd_function} transposed: {emd_function(image_ref_np.copy().T, image_pred_np.copy().T)} / patches: {emd_function(image_ref_np_patches.copy().T, image_pred_np_patches.copy().T)}"
        # )
        # mirror_ref = np.flip(image_ref_np.copy(), axis=1)
        # print(
        #     f"{emd_function} mirror: {emd_function(mirror_ref, image_ref_np.copy())} / patches: {emd_function(image_to_patches(mirror_ref), image_pred_np_patches.copy())}"
        # )
