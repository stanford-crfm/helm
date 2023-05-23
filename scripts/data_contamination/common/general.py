import json
import os
import shlex
import subprocess
import uuid
import zstandard
from typing import Any, Callable, Dict, List, Optional, TypeVar
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tqdm import tqdm

import pyhocon
from dataclasses import asdict, is_dataclass

from common.hierarchical_logger import hlog, htrack, htrack_block


def singleton(items: List):
    """Ensure there's only one item in `items` and return it."""
    if len(items) != 1:
        raise ValueError(f"Expected 1 item, got {len(items)} items: {items}")
    return items[0]


def flatten_list(ll: List):
    """
    Input: Nested lists
    Output: Flattened input
    """
    return sum(map(flatten_list, ll), []) if isinstance(ll, list) else [ll]


def ensure_directory_exists(path: str):
    """Create `path` if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def parse_hocon(text: str):
    """Parse `text` (in HOCON format) into a dict-like object."""
    return pyhocon.ConfigFactory.parse_string(text)


def shell(args: List[str]):
    """Executes the shell command in `args`."""
    cmd = shlex.join(args)
    hlog(f"Executing: {cmd}")
    exit_code = subprocess.call(args)
    if exit_code != 0:
        hlog(f"Failed with exit code {exit_code}: {cmd}")


def format_text(text: str) -> str:
    return json.dumps(text)


def format_text_lines(text: str) -> List[str]:
    return text.split("\n")


def format_tags(tags: List[str]) -> str:
    """Takes a list of tags and outputs a string: tag_1,tag_2,...,tag_n"""
    return f"[{','.join(tags)}]"


def format_split(split: str) -> str:
    """Format split"""
    return f"|{split}|"


def asdict_without_nones(obj: Any) -> Dict[str, Any]:
    if not is_dataclass(obj):
        raise ValueError(f"Expected dataclass, got '{obj}'")
    return asdict(obj, dict_factory=lambda x: {k: v for (k, v) in x if v is not None})


def binarize_dict(d: Dict[str, int]) -> Dict[str, int]:
    """Binarize the dict by setting the values that are 1 to 0.

    Values greater than 1 stay untouched.
    """
    return {k: 0 if v == 1 else v for k, v in d.items()}


def serialize(obj: Any) -> List[str]:
    """Takes in a dataclass and outputs all of its fields and values in a list."""
    if not is_dataclass(obj):
        raise ValueError(f"Expected dataclass, got '{obj}'")
    return [f"{key}: {json.dumps(value)}" for key, value in asdict(obj).items()]


def write(file_path: str, content: str):
    """Write content out to a file at path file_path."""
    hlog(f"Writing {len(content)} characters to {file_path}")
    with open(file_path, "w") as f:
        f.write(content)


def write_lines(file_path: str, lines: List[str]):
    """Write lines out to a file at path file_path."""
    hlog(f"Writing {len(lines)} lines to {file_path}")
    with open(file_path, "w") as f:
        for line in lines:
            print(line, file=f)


def indent_lines(lines: List[str], count: int = 2) -> List[str]:
    """Add `count` spaces before each line in `lines`."""
    prefix = " " * count
    return [prefix + line if len(line) > 0 else "" for line in lines]


def match_case(source_word: str, target_word: str) -> str:
    """Return a version of the target_word where the case matches the source_word."""
    # Check for all lower case source_word
    if all(letter.islower() for letter in source_word):
        return target_word.lower()
    # Check for all caps source_word
    if all(letter.isupper() for letter in source_word):
        return target_word.upper()
    # Check for capital source_word
    if source_word and source_word[0].isupper():
        return target_word.capitalize()
    return target_word


InT = TypeVar("InT")
OutT = TypeVar("OutT")


def parallel_map(
    process: Callable[[InT], OutT], items: List[InT], parallelism: int, multiprocessing: bool = False
) -> List[OutT]:
    """
    A wrapper for applying `process` to all `items`.
    """
    units = "processes" if multiprocessing else "threads"
    with htrack_block(f"Parallelizing computation on {len(items)} items over {parallelism} {units}"):
        results: List
        if parallelism == 1:
            results = list(tqdm(map(process, items), total=len(items), disable=None))
        elif multiprocessing:
            with ProcessPoolExecutor(max_workers=parallelism) as executor:
                results = list(tqdm(executor.map(process, items), total=len(items), disable=None))
        else:
            with ThreadPoolExecutor(max_workers=parallelism) as executor:
                results = list(tqdm(executor.map(process, items), total=len(items), disable=None))
    return results


def without_common_entries(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Given `items` (a list of dictionaries), return a corresponding list of
    dictionaries where all the common entries have been removed.
    """
    common_keys = [key for key in items[0] if all(item[key] == items[0][key] for item in items)]
    return [dict((key, value) for key, value in item.items() if key not in common_keys) for item in items]


def generate_unique_id() -> str:
    """
    Generate a unique ID (e.g., 77437ea482144bf7b9275a0acee997db).
    """
    return uuid.uuid4().hex


def get_file_name(path: str) -> str:
    """
    Get the file name from a path (e.g., /path/to/image.png => image.png).
    """
    return os.path.split(path)[-1]

