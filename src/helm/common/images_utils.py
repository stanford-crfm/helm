import base64
import io
import requests
import shutil
from typing import Optional

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
