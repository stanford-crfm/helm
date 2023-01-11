from typing import Callable

from .file_cache import FileCache


class GCSFileCache(FileCache):
    """
    To use GCS to store files.
    TODO: support in the future for an external storage.
    """

    def __init__(self, location: str, file_extension: str, binary_mode: bool = True):
        super().__init__(location, file_extension, binary_mode)

    def store(self, compute: Callable) -> str:
        raise NotImplementedError
