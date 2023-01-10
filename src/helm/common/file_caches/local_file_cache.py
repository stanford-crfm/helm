import os
from typing import Callable

from helm.common.general import ensure_directory_exists, generate_unique_id
from .file_cache import FileCache


class LocalFileCache(FileCache):
    def __init__(self, base_path: str, file_extension: str, binary_mode: bool = True):
        super().__init__(base_path, file_extension, binary_mode)
        ensure_directory_exists(base_path)

    def store(self, compute: Callable) -> str:
        """
        Stores the output of `compute` as a file at a unique path.
        Returns the file path.
        """
        file_path: str = self._generate_unique_file_path()
        with open(file_path, "wb" if self.binary_mode else "w") as f:
            f.write(compute())

        return file_path

    def _generate_unique_file_path(self) -> str:
        """Generate an unique file name at `base_path`"""

        def generate_one() -> str:
            file_name: str = f"{generate_unique_id()}.{self.file_extension}"
            return os.path.join(self.location, file_name)

        file_path: str
        while True:
            file_path = generate_one()
            if not os.path.exists(file_path):
                break
        return file_path
