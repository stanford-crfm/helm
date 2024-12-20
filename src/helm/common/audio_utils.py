from io import BytesIO
import os
from typing import Optional
from filelock import FileLock

import numpy as np
import soundfile as sf
import subprocess

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


def use_ffmpeg_to_convert_audio_file(input_path: str, output_path: str) -> None:
    if os.path.exists(output_path):
        return
    """Use ffmpeg to convert an audio file type"""
    try:
        subprocess.run(["ffmpeg", "-i", input_path, output_path], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise ValueError("Please install ffmpeg using `bash install-shelm-extras.sh` first to convert audio files.")


def use_ffmpeg_to_extract_audio_from_video(input_video_path: str, output_audio_path: str) -> None:
    if os.path.exists(output_audio_path):
        return
    try:
        subprocess.run(["ffmpeg", "-i", input_video_path, "-q:a", "0", "-map", "a", output_audio_path], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise ValueError("Please install ffmpeg using `bash install-shelm-extras.sh` first to extract audio files.")
