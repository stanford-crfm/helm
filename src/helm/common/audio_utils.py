from io import BytesIO
from typing import Optional
from filelock import FileLock
import base64
import os

import ffmpeg
import numpy as np
import soundfile as sf
import subprocess

from helm.common.hierarchical_logger import hlog
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


def is_invalid_audio_file(audio_path: str) -> bool:
    """
    Two conditions for an audio file to be considered invalid:
    1. The file does not exist.
    2. The file is empty.
    """
    if not os.path.exists(audio_path):
        return True

    try:
        with sf.SoundFile(audio_path) as audio_file:
            return len(audio_file) == 0
    except RuntimeError:
        return True


def extract_audio(video_path: str, output_audio_path: str) -> None:
    """
    Extracts audio from an MP4 video file and saves it as an MP3 file.

    Args:
        video_path (str): Path to the input MP4 video file.
        output_audio_path (str): Path to save the extracted MP3 audio file.

    Returns:
        None
    """
    try:
        (
            ffmpeg.input(video_path)
            .output(output_audio_path, format="mp3", acodec="libmp3lame", audio_bitrate="192k")
            .run(overwrite_output=True)
        )
    except ffmpeg.Error as e:
        hlog(f"Error extracting audio from video: {video_path}: {e.stderr.decode()}")
        raise e


def encode_audio_to_base64(file_path: str) -> str:
    """
    Encodes an audio file to a Base64 string.

    Args:
        file_path (str): Path to the audio file.

    Returns:
        str: Base64-encoded string of the audio file.
    """
    assert os.path.exists(file_path), f"Audio file does not exist at path: {file_path}"
    with open(file_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode("utf-8")
