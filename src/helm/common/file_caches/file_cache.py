from abc import ABC, abstractmethod
from typing import Callable


class FileCache(ABC):
    """
    Cache to store files.

    Attributes
    ----------
    location: str
        Location of the files (e.g., local path to a directory or a URL)
    file_extension: str
        The types of file we're storing in the cache (e.g., png, jpg, mp4, etc.).
    binary_mode: bool
        Whether to write in binary mode (default is True).
    """

    def __init__(self, location: str, file_extension: str, binary_mode: bool = True):
        self.location: str = location
        self.file_extension: str = file_extension
        self.binary_mode: bool = binary_mode

    @abstractmethod
    def store(self, compute: Callable) -> str:
        """
        Stores the output of `compute` as a file at a unique location.
        Returns the location of the file.
        """
        pass
