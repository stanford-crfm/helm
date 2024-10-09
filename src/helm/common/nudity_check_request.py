from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass(frozen=True)
class NudityCheckRequest:
    """
    Checks for nudity for a given set of images.
    """

    # Batch of images
    image_locations: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class NudityCheckResult:
    """Result after sending a `NudityCheckRequest`."""

    # Whether the request was successful
    success: bool

    # Whether the request was cached
    cached: bool

    # Nudity results. True indicates the particular image contains nudity.
    image_to_nudity: Dict[str, bool] = field(default_factory=dict)

    # If `success` is false, what was the error?
    error: Optional[str] = None
