import os
from typing import List
import pyhocon


def singleton(items: List):
    """Ensure there's only one item in `items` and return it."""
    if len(items) != 1:
        raise ValueError(f"Expected 1 item, got {len(items)}")
    return items[0]


def ensure_directory_exists(path: str):
    """Create `path` if it doesn't exist."""
    if not os.path.exists(path):
        os.mkdir(path)


def parse_hocon(text: str):
    """Parse `text` (in HOCON format) into a dict-like object."""
    return pyhocon.ConfigFactory.parse_string(text)
