import os
from filelock import FileLock

import numpy as np
from scipy.io import wavfile

from helm.common.optional_dependencies import handle_module_not_found_error


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
