from io import BytesIO
import os
from typing import Optional
from filelock import FileLock

import numpy as np
import soundfile as sf

from helm.common.multimodal_request_utils import get_contents_as_bytes
from helm.common.optional_dependencies import handle_module_not_found_error

try:
    import librosa
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, ["audiolm"])


def ensure_audio_file_exists_from_array(path: str, array: np.ndarray, sample_rate: int) -> None:
    """Write the array to the wav or mp3 file if it does not already exist.

    Uses file locking and an atomic rename to avoid file corruption due to incomplete writes and
    concurrent writes."""
    file_extension = os.path.splitext(path)[1]
    if file_extension != ".wav" and file_extension != ".mp3":
        raise ValueError(f"Path must end with .wav or .mp3: {path}")
    with FileLock(f"{path}.lock"):
        if os.path.exists(path):
            # Skip because file already exists
            return
        path_prefix = path.removesuffix(file_extension)
        tmp_path = f"{path_prefix}.tmp{file_extension}"
        sf.write(tmp_path, array, samplerate=sample_rate)
        os.rename(tmp_path, path)


def get_array_from_audio_file(path: str, sample_rate: Optional[int]) -> np.ndarray:
    """Get an array from an audio file"""
    audio_file = (
        BytesIO(get_contents_as_bytes(path)) if path.startswith("http://") or path.startswith("https://") else path
    )

    # librosa accepts a local file path or a file-like object
    audio_array, _ = librosa.load(audio_file, sr=sample_rate)
    return audio_array
