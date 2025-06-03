from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class VideoUnderstandingParameters:
    """
    Parameters for video understanding.
    """

    sample_frames: bool = False
    """Sample image frames from the video instead of using the video directly."""

    frames_per_second: int = 1
    """Number of frames per second to sample from the video."""


