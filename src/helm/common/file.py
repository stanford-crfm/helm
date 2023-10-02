import os
import urllib
from dataclasses import dataclass


# Supported media types
IMAGE_MEDIA_TYPE: str = "image"


@dataclass(frozen=True)
class File:
    """Represents a file."""

    location: str
    """The location of the file, which can be a local path or URL"""

    media_type: str
    """The type of the media (e.g., image, audio, etc.)"""

    extension: str
    """The file extension"""

    is_compressed: bool = False
    """Whether the file is compressed"""

    @property
    def is_local_file(self) -> bool:
        """Returns `True` if the file is a local file and False if `path` is a URL."""
        return urllib.parse.urlparse(self.location).scheme not in ["http", "https"]

    def __post_init__(self):
        if self.is_local_file:
            assert os.path.exists(self.location), f"Local file does not exist at path: {self.location}"
