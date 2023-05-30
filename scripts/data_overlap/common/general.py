from typing import Any, Dict

from dataclasses import asdict, is_dataclass
from common.hierarchical_logger import hlog


def asdict_without_nones(obj: Any) -> Dict[str, Any]:
    if not is_dataclass(obj):
        raise ValueError(f"Expected dataclass, got '{obj}'")
    return asdict(obj, dict_factory=lambda x: {k: v for (k, v) in x if v is not None})


def write(file_path: str, content: str):
    """Write content out to a file at path file_path."""
    hlog(f"Writing {len(content)} characters to {file_path}")
    with open(file_path, "w") as f:
        f.write(content)
