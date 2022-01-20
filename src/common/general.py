import os
import shlex
import subprocess
from typing import List

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
def ensure_file_downloaded(source_url: str, target_path: str, untar: bool = False):
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
    if untar:
        tmp2_path = target_path + ".tmp2"
        ensure_directory_exists(tmp2_path)
        shell(["tar", "xf", tmp_path, "-C", tmp2_path])
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


def format_tags(tags: List[str]) -> str:
    """Takes a list of tags and outputs a string: [tag_1,tag_2,...,tag_n]."""
    return f"[{','.join(tags)}]"


def serialize(obj: dataclass) -> List[str]:
    """Takes in a dataclass and outputs all of its fields and values in a list."""
    return [f"{key}: {value}" for key, value in asdict(obj).items()]


def write(file_path: str, content: str):
    """Write content out to a file at path file_path."""
    with open(file_path, "w") as f:
        f.write(content)
