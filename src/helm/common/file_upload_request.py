from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class FileUploadRequest:
    """Uploads a file at `path`."""

    # Path of the file to upload
    path: str


@dataclass(frozen=True)
class FileUploadResult:
    """Result after sending a `FileUploadRequest`."""

    # Whether the request was successful
    success: bool

    # Whether the request was cached
    cached: bool

    # URL of the uploaded file
    url: str

    # If `success` is false, what was the error?
    error: Optional[str] = None
