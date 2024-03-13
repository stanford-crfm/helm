from typing import List, Tuple

from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from toolbox.printing import debug
from tqdm import tqdm

import numpy as np
import math


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

    return most_frequent_color, frequency


def img_to_sig_patches(
    img: np.array,
    rgb_most_frequent_color: np.array,
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
    img = np.mean(img, axis=2, keepdims=True)

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
    gray_most_frequent_color: float = np.mean(rgb_most_frequent_color) / 255.0
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
    sub_sigs: np.array,
    patch_size: Tuple[int, int],
    gray_most_frequent_color: float,
    weight_most_frequent_color: float = 0.01,
) -> np.array:
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
    sig1: np.array,
    sig2: np.array,
    gray_most_frequent_color: float,
    patch_size: Tuple[int, int],
    weight_most_frequent_color: float = 0.01,
    use_tqdm: bool = True,
) -> np.array:
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
    for i in tqdm(range(sig1.shape[0]), disable=not use_tqdm):
        for j in range(sig2.shape[0]):
            pos_sig1 = sig1[i, -2:]
            pos_sig2 = sig2[j, -2:]
            # debug(np.sum(sig1_reshaped[i, 0]))
            emd_value, _, _ = cv2.EMD(sig1_reshaped[i], sig2_reshaped[j], cv2.DIST_L2)
            cost_matrix[i, j] = emd_value + patch_size[0] * np.linalg.norm(pos_sig1 - pos_sig2)
    return cost_matrix.astype(np.float32)


def compute_emd(
    img1_PIL: Image.Image,
    img2_PIL: Image.Image,
    threshold: float = 0.5,
    patch_size: Tuple[int, int] = (8, 8),
    max_num_patches: int = 100,
    weight_most_frequent_color: float = 0.01,
    use_tqdm: bool = True,
):
    assert img1_PIL.size == img2_PIL.size

    # Resize the images so that there are not too many patches
    # Try to maintain the aspect ratio and resize to a multiple of the patch size
    num_patches = math.ceil(img1_PIL.size[0] / patch_size[0]) * math.ceil(img1_PIL.size[1] / patch_size[1])
    if num_patches > max_num_patches:
        ideal_divider = (num_patches / max_num_patches) ** 0.5
        closest_round_width = math.ceil((img1_PIL.size[0] / patch_size[1]) / ideal_divider) * patch_size[1]
        num_patches_width = closest_round_width / patch_size[0]
        # Chooses a round height such that:
        # - (round_width / patch_size[1]) * (round_height / patch_size[0]) <= max_num_patches
        # - the ratio is as unchanged as possible: (original_width / round_width) / (original_height / round_height) is close to 1
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
    (rgb_most_frequent_color, frequency) = get_most_frequent_color(img1_np)
    gray_most_frequent_color = np.mean(rgb_most_frequent_color) / 255.0
    sig1 = img_to_sig_patches(img1_np, rgb_most_frequent_color, patch_size, weight_most_frequent_color)
    sig2 = img_to_sig_patches(img2_np, rgb_most_frequent_color, patch_size, weight_most_frequent_color)

    if frequency > threshold:
        # Ignore patches that are constant equal to the most frequent color
        mask1 = np.any(sig1[:, 1:-2] != gray_most_frequent_color, axis=1)
        mask2 = np.any(sig2[:, 1:-2] != gray_most_frequent_color, axis=1)
        mask = np.logical_or(mask1, mask2)
        sig1 = sig1[mask]
        sig2 = sig2[mask]

    # Normalize the weights
    weight1 = sig1[:, 0]
    weight2 = sig2[:, 0]
    sig1[:, 0] /= np.sum(weight1)
    sig2[:, 0] /= np.sum(weight2)
    # weights = np.maximum(weight1, weight2)
    # weights /= np.sum(weights)
    # sig1[:, 0] = weights
    # sig2[:, 0] = weights

    # Compute EMD
    cost = compute_cost_matrix_on_sig(
        sig1=sig1,
        sig2=sig2,
        gray_most_frequent_color=gray_most_frequent_color,
        patch_size=patch_size,
        weight_most_frequent_color=weight_most_frequent_color,
        use_tqdm=use_tqdm,
    )
    emd_value, _, flow = cv2.EMD(sig1, sig2, cv2.DIST_USER, cost)

    plot_flow(
        sig1,
        sig2,
        flow=flow,
        dim=(closest_round_height, closest_round_width),
        patch_size=patch_size,
        most_frequent_color=gray_most_frequent_color,
    )

    return emd_value


def plot_flow_simple(sig1, sig2, flow, dim, arrow_width_scale=3):
    """Plots the flow computed by cv2.EMD

    The source images are retrieved from the signatures and
    plotted in a combined image, with the first image in the
    red channel and the second in the green. Arrows are
    overplotted to show moved earth, with arrow thickness
    indicating the amount of moved earth."""

    img1 = sig_to_img(sig1, dim)
    img2 = sig_to_img(sig2, dim)
    combined = np.dstack((img1, img2, 0 * img2))
    # RGB values should be between 0 and 1
    combined /= combined.max()
    print('Red channel is "before"; green channel is "after"; yellow means "unchanged"')
    plt.imshow(combined)

    flows = np.transpose(np.nonzero(flow))
    for src, dest in flows:
        # Skip the pixel value in the first element, grab the
        # coordinates. It'll be useful later to transpose x/y.
        start = sig1[src, -2:][::-1]
        end = sig2[dest, -2:][::-1]
        if np.all(start == end):
            # Unmoved earth shows up as a "flow" from a pixel
            # to that same exact pixel---don't plot mini arrows
            # for those pixels
            continue
        start = start * dim[::-1]
        end = end * dim[::-1]

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
            color="purple",
            edgecolor="purple",
            linewidth=mag / 3,
            width=mag,
            units="dots",
            headlength=5,
            headwidth=3,
            headaxislength=4.5,
        )

    plt.title("Earth moved from img1 to img2")


def plot_flow(sig1, sig2, flow, dim, patch_size, most_frequent_color, arrow_width_scale=3):
    """Plots the flow computed by cv2.EMD

    The source images are retrieved from the signatures and
    plotted in a combined image, with the first image in the
    red channel and the second in the green. Arrows are
    overplotted to show moved earth, with arrow thickness
    indicating the amount of moved earth."""

    img1 = patch_sig_to_img(sig1, dim, patch_size, most_frequent_color=most_frequent_color)
    img2 = patch_sig_to_img(sig2, dim, patch_size, most_frequent_color=most_frequent_color)
    combined = np.dstack((img1, img2, 0 * img2))
    # RGB values should be between 0 and 1
    combined /= combined.max()
    print('Red channel is "before"; green channel is "after"; yellow means "unchanged"')
    plt.imshow(combined)

    flows = np.transpose(np.nonzero(flow))
    for src, dest in flows:
        # Skip the pixel value in the first element, grab the
        # coordinates. It'll be useful later to transpose x/y.
        start = sig1[src, -2:][::-1]
        end = sig2[dest, -2:][::-1]
        if np.all(start == end):
            # Unmoved earth shows up as a "flow" from a pixel
            # to that same exact pixel---don't plot mini arrows
            # for those pixels
            continue
        start = start * dim[::-1]
        end = end * dim[::-1]

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
            color="purple",
            edgecolor="purple",
            linewidth=mag * patch_size[0] / 3,
            width=mag * patch_size[0],
            units="dots",
            headlength=5 * patch_size[0] / 3,
            headwidth=3 * patch_size[0] / 3,
            headaxislength=4.5 * patch_size[0] / 3,
        )

    plt.title("Earth moved from img1 to img2")
    plt.show()


def sig_to_img(sig, dim):
    """Convert a signature back to a 2D image"""
    img = np.zeros(dim, dtype=float)
    for i in range(sig.shape[0]):
        x = round(sig[i, 3] * dim[1])
        y = round(sig[i, 2] * dim[0])
        img[y, x] = sig[i, 1]
    return img


def patch_sig_to_img(sig, dim, patch_size, most_frequent_color):
    """Convert a signature back to a 2D image"""
    patches_sig = reshape_sub_sig_batch(sig[:, :-2], patch_size, np.array([1.0]))
    img = np.ones(dim, dtype=float) * most_frequent_color
    for i in range(patches_sig.shape[0]):
        xx = round(dim[1] * sig[i, -1])
        yy = round(dim[0] * sig[i, -2])
        for j in range(patches_sig.shape[1]):
            y = round(patches_sig[i, j, 3] * patch_size[1])
            x = round(patches_sig[i, j, 2] * patch_size[0])
            img[y + yy, x + xx] = patches_sig[i, j, 1]
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
    num: int = 1
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
