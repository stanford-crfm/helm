import base64
import os
from scipy.io.wavfile import write
import numpy as np


def encode_base64(audio_path: str) -> str:
    """Returns the base64 representation of an audio file."""
    with open(audio_path, "rb") as audio_file:
        audio_data = audio_file.read()
        return base64.b64encode(audio_data).decode("utf-8")


def ensure_audio_file_exists(audio_path: str, audio_array: np.ndarray, audio_sampling_rate: int) -> None:
    """Ensures that the audio file exists locally."""
    if not os.path.exists(audio_path):
        write(audio_path, audio_sampling_rate, audio_array)
