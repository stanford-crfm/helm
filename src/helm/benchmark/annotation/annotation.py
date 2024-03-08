from typing import Any
from dataclasses import dataclass


@dataclass
class Annotation:
    data: Any
    """Data of the annotation"""

    displayable: bool = False
    """Whether it should be displayed in the frontend or not."""
