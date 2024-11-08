from io import BytesIO
import os
from typing import Optional
from filelock import FileLock

import librosa
import numpy as np
from scipy.io import wavfile

from helm.common.multimodal_request_utils import get_contents_as_bytes


def ensure_wav_file_exists_from_array(path: str, array: np.ndarray, sample_rate: int) -> None:
    """Write the array to the wav file if it does not already exist.

    Uses file locking and an atomic rename to avoid file corruption due to incomplete writes and
    concurrent writes."""
    if not path.endswith(".wav"):
        raise ValueError(f"Path must end with .wav: {path}")
    with FileLock(f"{path}.lock"):
        if os.path.exists(path):
            # Skip because file already exists
            return
        path_prefix = path.removesuffix(".wav")
        tmp_path = f"{path_prefix}.tmp.wav"
        wavfile.write(tmp_path, array, samplerate=sample_rate)
        os.rename(tmp_path, path)


def ensure_mp3_file_exists_from_array(path: str, array: np.ndarray, sample_rate: int) -> None:
    """Write the array to the mp3 file if it does not already exist.

    Uses file locking and an atomic rename to avoid file corruption due to incomplete writes and
    concurrent writes."""
    if not path.endswith(".mp3"):
        raise ValueError(f"Path must end with .mp3: {path}")
    with FileLock(f"{path}.lock"):
        if os.path.exists(path):
            # Skip because file already exists
            return
        path_prefix = path.removesuffix(".mp3")
        tmp_path = f"{path_prefix}.tmp.mp3"
        wavfile.write(tmp_path, array, samplerate=sample_rate)
        os.rename(tmp_path, path)


def get_array_from_audio_file(path: str, sample_rate: Optional[int]) -> None:
    """Get an array from an audio file"""
    audio_file = (
        BytesIO(get_contents_as_bytes(path)) if path.startswith("http://") or path.startswith("https://")
        else path)
    # librosa accepts a local file path or a file-like object
    audio_array, _ = librosa.load(audio_file, sr=sample_rate)
    return audio_array
