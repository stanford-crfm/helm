import json
import os
import shlex
import subprocess
from typing import List, Optional

import pyhocon
from dataclasses import asdict, dataclass

from common.hierarchical_logger import htrack, hlog


def singleton(items: List):
    """Ensure there's only one item in `items` and return it."""
    if len(items) != 1:
        raise ValueError(f"Expected 1 item, got {len(items)}")
    return items[0]


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
    tmp_path = target_path + ".tmp"
    # TODO: -c doesn't work for some URLs
    #       https://github.com/stanford-crfm/benchmarking/issues/51
    shell(["wget", source_url, "-O", tmp_path])

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
            shell(["unzstd", tmp_path, "-o", tmp2_path])
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
        shell(["mv", tmp_path, target_path])
    hlog(f"Finished downloading {source_url} to {target_path}")


def format_text(text: str) -> str:
    return json.dumps(text)


def format_text_lines(text: str) -> List[str]:
    return text.split("\n")


def format_tags(tags: List[str]) -> str:
    """Takes a list of tags and outputs a string: tag_1,tag_2,...,tag_n"""
    return f"[{','.join(tags)}]"


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
