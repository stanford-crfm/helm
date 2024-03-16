from abc import ABC, abstractmethod
from typing import Callable


class FileCache(ABC):
    """
    Cache to store files.
    """

    @abstractmethod
    def store(self, compute: Callable[[], bytes]) -> str:
        """
        Stores the output of `compute` as a file at a unique location.
        Returns the location of the file.
        """
        pass
