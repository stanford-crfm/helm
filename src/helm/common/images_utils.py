from hashlib import md5
import base64
import io
import os
import math
import requests
import shutil
from pathlib import Path
from typing import List, Optional, Tuple, Union
from urllib.request import urlopen

import numpy as np

from helm.common.general import is_url
from helm.common.optional_dependencies import handle_module_not_found_error

try:
    from PIL import Image
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, ["images"])

try:
    import cv2
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, ["vlm"])


def open_image(image_location: str) -> Image.Image:
    """
    Opens image with the Python Imaging Library.
    """
    image: Image.Image
    if is_url(image_location):
        image = Image.open(requests.get(image_location, stream=True).raw)  # type: ignore
    else:
        image = Image.open(image_location)
    return image.convert("RGB")


def get_dimensions(image_location: str) -> Tuple[int, int]:
    """Returns the dimensions of the image."""
    image: Image.Image = open_image(image_location)
    return image.size


def encode_base64(image_location: str, format="JPEG") -> str:
    """Returns the base64 representation of an image file."""
    image_file = io.BytesIO()
    image: Image.Image = open_image(image_location)
    image.save(image_file, format=format)
    return base64.b64encode(image_file.getvalue()).decode("ascii")


def generate_hash(image: Image.Image) -> str:
    """Generates a hash for the image."""
    return md5(image.tobytes()).hexdigest()


def copy_image(src: str, dest: str, width: Optional[int] = None, height: Optional[int] = None) -> None:
    """
    Copies the image file from `src` path to `dest` path. If dimensions `width` and `height`
    are specified, resizes the image before copying. `src` can be a URL.
    """
    if (width is not None and height is not None) or is_url(src):
        image = open_image(src)
        if width is not None and height is not None:
            image = image.resize((width, height), Image.Resampling.LANCZOS)
        image.save(dest)
    else:
        shutil.copy(src, dest)


def resize_image_to_max_file_size(src: str, dest: str, max_size_in_bytes: int, step=10):
    # Open an image file
    with Image.open(src) as img:
        width, height = img.size

        # Reduce dimensions iteratively until the file size is under the limit
        while True:
            # Save the image temporarily to check the file size
            img.save(dest, quality=95)  # Start with high quality
            if os.path.getsize(dest) < max_size_in_bytes:
                break

            # Reduce dimensions
            width -= step
            height -= step
            img = img.resize((width, height), Image.Resampling.LANCZOS)


def is_blacked_out_image(image_location: str) -> bool:
    """Returns True if the image is all black. False otherwise."""
    if is_url(image_location):
        arr = np.asarray(bytearray(urlopen(image_location).read()), dtype=np.uint8)
        image = cv2.imdecode(arr, -1)
    else:
        image = cv2.imread(image_location, 0)
    return cv2.countNonZero(image) == 0


def filter_blacked_out_images(image_locations: List[str]) -> List[str]:
    """Returns a list of image locations that are not blacked out."""
    return [image_location for image_location in image_locations if not is_blacked_out_image(image_location)]


def sample_frames(
    video_path: Union[str, Path],
    fps: int
) -> List[str]:
    """
    Open the video at `video_path`, sample frames at approximately `fps` frames/sec,
    save each sampled frame as a JPEG under:
        <video_folder>/<video_name_without_ext>/frames/

    If that folder already exists and contains at least the expected number of JPEGs, skip resampling
    and return the existing image paths.

    Returns a list of all saved image‐file paths (as strings).
    """
    # Allow either a str or Path; normalize to Path for filesystem operations
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Determine output folder: same parent, named after the video, with a "frames" subfolder.
    video_folder = video_path.parent
    video_stem = video_path.stem
    output_dir = video_folder / "frames" / video_stem

    # Try to open the video to compute expected number of sampled frames
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0

    # Compute duration (in seconds). Guard against division by zero.
    duration_sec = total_frames / native_fps if native_fps > 0 else 0
    expected_count = int(math.ceil(duration_sec * fps))

    # If frames folder exists and has the expected number of JPEGs, return existing
    if output_dir.exists():
        existing = sorted(output_dir.glob("*.jpg"))
        if len(existing) >= expected_count and expected_count > 0:
            cap.release()
            return [str(p) for p in existing]

        # Otherwise, clear and recreate
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    interval = max(int(round(native_fps / fps)), 1)
    frame_idx = 0
    saved_idx = 0
    saved_paths: List[str] = []

    while True:
        grabbed = cap.grab()
        if not grabbed:
            break

        if frame_idx % interval == 0:
            success_decode, frame = cap.retrieve()
            if not success_decode:
                break

            img_filename = f"frame_{saved_idx:04d}.jpg"
            img_path = output_dir / img_filename
            cv2.imwrite(str(img_path), frame)
            saved_paths.append(str(img_path))
            saved_idx += 1

        frame_idx += 1

    cap.release()
    return saved_paths
