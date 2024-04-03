import base64
import io
import os

import requests
import shutil
import tempfile
from typing import List, Optional
from urllib.request import urlopen

import numpy as np

from .general import is_url
from helm.common.optional_dependencies import handle_module_not_found_error

try:
    from PIL import Image
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, ["images"])


def open_image(image_location: str) -> Image.Image:
    """
    Opens image with the Python Imaging Library.
    """
    image: Image.Image
    if is_url(image_location):
        image = Image.open(requests.get(image_location, stream=True).raw)
    else:
        image = Image.open(image_location)
    return image.convert("RGB")


def encode_base64(image_location: str, format="JPEG") -> str:
    """Returns the base64 representation of an image file."""
    image_file = io.BytesIO()
    image: Image.Image = open_image(image_location)
    image.save(image_file, format=format)
    return base64.b64encode(image_file.getvalue()).decode("ascii")


def fit_within_max_dimension(image_path: str, max_dimension: int, output_path: Optional[str] = None) -> str:
    """
    Resizes an image so that neither dimension exceeds max_dimension.
    If the image is already smaller than max_dimension, it is not resized and the original path is returned.

    Parameters:
    - image: The image to be resized.
    - max_dimension: The maximum allowed size for the image's width and height.

    Returns:
    - The path to the resized image.
    """
    with Image.open(image_path) as image:
        width, height = image.size
        scaling_factor = min(max_dimension / width, max_dimension / height)
        if scaling_factor < 1:
            new_width = int(width * scaling_factor)
            new_height = int(height * scaling_factor)
            image = image.resize((new_width, new_height), Image.ANTIALIAS)
        else:
            return image_path

        if output_path:
            image.save(output_path)
        else:
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                image.save(temp_file)
                output_path = temp_file.name

        assert os.path.exists(output_path)
        return output_path


def copy_image(src: str, dest: str, width: Optional[int] = None, height: Optional[int] = None):
    """
    Copies the image file from `src` path to `dest` path. If dimensions `width` and `height`
    are specified, resizes the image before copying. `src` can be a URL.
    """
    if (width is not None and height is not None) or is_url(src):
        image = open_image(src)
        if width is not None and height is not None:
            image = image.resize((width, height), Image.ANTIALIAS)
        image.save(dest)
    else:
        shutil.copy(src, dest)


def is_blacked_out_image(image_location: str) -> bool:
    """Returns True if the image is all black. False otherwise."""
    try:
        import cv2
    except ModuleNotFoundError as e:
        handle_module_not_found_error(e, ["heim"])

    if is_url(image_location):
        arr = np.asarray(bytearray(urlopen(image_location).read()), dtype=np.uint8)
        image = cv2.imdecode(arr, -1)
    else:
        image = cv2.imread(image_location, 0)
    return cv2.countNonZero(image) == 0


def filter_blacked_out_images(image_locations: List[str]) -> List[str]:
    """Returns a list of image locations that are not blacked out."""
    return [image_location for image_location in image_locations if not is_blacked_out_image(image_location)]
