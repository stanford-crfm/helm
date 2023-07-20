import base64
import io
import requests
import shutil
from urllib.request import urlopen
from typing import Optional, List

import numpy as np
from PIL import Image

from .general import is_url


def is_blacked_out_image(image_location: str) -> bool:
    """Returns True if the image is all black. False otherwise."""
    import cv2

    if is_url(image_location):
        arr = np.asarray(bytearray(urlopen(image_location).read()), dtype=np.uint8)
        image = cv2.imdecode(arr, -1)
    else:
        image = cv2.imread(image_location, 0)
    return cv2.countNonZero(image) == 0


def filter_blacked_out_images(image_locations: List[str]) -> List[str]:
    """Returns a list of image locations that are not blacked out."""
    return [image_location for image_location in image_locations if not is_blacked_out_image(image_location)]


def open_image(image_location: str) -> Image:
    """
    Opens image with the Python Imaging Library.
    """
    image: Image
    if is_url(image_location):
        image = Image.open(requests.get(image_location, stream=True).raw)
    else:
        image = Image.open(image_location)
    return image.convert("RGB")


def encode_base64(image_location: str) -> str:
    """Returns the base64 representation of an image file."""
    image_file = io.BytesIO()
    image: Image = open_image(image_location)
    image.save(image_file, format="JPEG")
    return base64.b64encode(image_file.getvalue()).decode("ascii")


def copy_image(src: str, dest: str, width: Optional[int] = None, height: Optional[int] = None):
    """
    Copies the image file from `src` path to `dest` path. If dimensions `width` and `height`
    are specified, resizes the image before copying. `src` can be a URL.
    """
    if (width is not None and height is not None) or is_url(src):
        image = open_image(src)
        resized_image = image.resize((width, height), Image.ANTIALIAS)
        resized_image.save(dest)
    else:
        shutil.copy(src, dest)
