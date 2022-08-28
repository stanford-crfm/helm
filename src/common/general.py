import json
import os
import shlex
import subprocess
import zstandard
from typing import Any, Callable, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tqdm import tqdm

import pyhocon
from dataclasses import asdict, dataclass

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


@htrack(None)
def ensure_file_downloaded(source_url: str, target_path: str, unpack: bool = False, unpack_type: Optional[str] = None):
    """Download `source_url` to `target_path` if it doesn't exist."""
    if os.path.exists(target_path):
        # Assume it's all good
        hlog(f"Not downloading {source_url} because {target_path} already exists")
        return

    # Download
    # gdown is used to download large files/zip folders from Google Drive.
    # It bypasses security warnings which wget cannot handle.
    downloader_executable: str = "gdown" if source_url.startswith("https://drive.google.com") else "wget"
    tmp_path: str = f"{target_path}.tmp"
    shell([downloader_executable, source_url, "-O", tmp_path])

    # Unpack (if needed) and put it in the right location
    if unpack:
        if unpack_type is None:
            if source_url.endswith(".tar") or source_url.endswith(".tar.gz"):
                unpack_type = "untar"
            elif source_url.endswith(".zip"):
                unpack_type = "unzip"
            elif source_url.endswith(".zst"):
                unpack_type = "unzstd"
            else:
                raise Exception("Failed to infer the file format from source_url. Please specify unpack_type.")

        tmp2_path = target_path + ".tmp2"
        ensure_directory_exists(tmp2_path)
        if unpack_type == "untar":
            shell(["tar", "xf", tmp_path, "-C", tmp2_path])
        elif unpack_type == "unzip":
            shell(["unzip", tmp_path, "-d", tmp2_path])
        elif unpack_type == "unzstd":
            dctx = zstandard.ZstdDecompressor()
            with open(tmp_path, "rb") as ifh, open(os.path.join(tmp2_path, "data"), "wb") as ofh:
                dctx.copy_stream(ifh, ofh)
        else:
            raise Exception("Invalid unpack_type")
        files = os.listdir(tmp2_path)
        if len(files) == 1:
            # If contains one file, just get that one file
            shell(["mv", os.path.join(tmp2_path, files[0]), target_path])
            os.rmdir(tmp2_path)
        else:
            shell(["mv", tmp2_path, target_path])
        os.unlink(tmp_path)
    else:
        if source_url.endswith(".gz"):
            gzip_path = f"{target_path}.gz"
            shell(["mv", tmp_path, gzip_path])
            # gzip writes its output to a file named the same as the input file, omitting the .gz extension
            shell(["gzip", "-d", gzip_path])
        else:
            shell(["mv", tmp_path, target_path])
    hlog(f"Finished downloading {source_url} to {target_path}")


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


def asdict_without_nones(obj: dataclass) -> Dict[str, Any]:
    return asdict(obj, dict_factory=lambda x: {k: v for (k, v) in x if v is not None})


def serialize(obj: dataclass) -> List[str]:
    """Takes in a dataclass and outputs all of its fields and values in a list."""
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
    """ Return a version of the target_word where the case matches the source_word. """
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


def parallel_map(process: Callable[[List], List], items: List, parallelism: int, multiprocessing: bool) -> List:
    """
    A wrapper for applying `process` to all `items`.
    """
    units = "processes" if multiprocessing else "threads"
    multiprocessing = True
    with htrack_block(f"Parallelizing computation on {len(items)} items over {parallelism} {units}"):
        if multiprocessing:
            with ProcessPoolExecutor(max_workers=parallelism) as executor:
                results: List = list(tqdm(executor.map(process, items), total=len(items)))
        else:
            with ThreadPoolExecutor(max_workers=parallelism) as executor:
                results: List = list(tqdm(executor.map(process, items), total=len(items)))
    return results
