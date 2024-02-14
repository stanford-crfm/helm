from filelock import FileLock
import json
import os
import shlex
import subprocess
import urllib
import uuid
import zstandard
from typing import Any, Callable, Dict, List, Optional, TypeVar
from datetime import datetime, date
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

import pyhocon
from dataclasses import asdict, is_dataclass

from helm.common.hierarchical_logger import hlog, htrack, htrack_block
from helm.common.optional_dependencies import handle_module_not_found_error


_CREDENTIALS_FILE_NAME = "credentials.conf"
_CREDENTIALS_ENV_NAME = "HELM_CREDENTIALS"


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


def get_credentials(base_path: str = "prod_env") -> Dict[str, str]:
    print(f"Looking in path: {base_path}")
    raw_credentials = os.getenv(_CREDENTIALS_ENV_NAME, "")
    credentials_path = os.path.join(base_path, _CREDENTIALS_FILE_NAME)
    if not raw_credentials and os.path.exists(credentials_path):
        with open(credentials_path) as f:
            raw_credentials = f.read()
    return parse_hocon(raw_credentials)


def shell(args: List[str]):
    """Executes the shell command in `args`."""
    cmd = shlex.join(args)
    hlog(f"Executing: {cmd}")
    exit_code = subprocess.call(args)
    if exit_code != 0:
        raise Exception(f"Failed with exit code {exit_code}: {cmd}")


@htrack(None)
def ensure_file_downloaded(
    source_url: str,
    target_path: str,
    unpack: bool = False,
    downloader_executable: str = "wget",
    unpack_type: Optional[str] = None,
):
    """Download `source_url` to `target_path` if it doesn't exist."""
    with FileLock(f"{target_path}.lock"):
        if os.path.exists(target_path):
            # Assume it's all good
            hlog(f"Not downloading {source_url} because {target_path} already exists")
            return

        # Download
        # gdown is used to download large files/zip folders from Google Drive.
        # It bypasses security warnings which wget cannot handle.
        if source_url.startswith("https://drive.google.com"):
            try:
                import gdown  # noqa
            except ModuleNotFoundError as e:
                handle_module_not_found_error(e, ["scenarios"])
            downloader_executable = "gdown"
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
            # Don't decompress if desired `target_path` ends with `.gz`.
            if source_url.endswith(".gz") and not target_path.endswith(".gz"):
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


def asdict_without_nones(obj: Any) -> Dict[str, Any]:
    if not is_dataclass(obj):
        raise ValueError(f"Expected dataclass, got '{obj}'")
    return asdict(obj, dict_factory=lambda x: {k: v for (k, v) in x if v is not None})


def serialize_dates(obj):
    """Serialize dates (pass deault=serialize_dates into json.dumps)."""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} is not serializable")


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


def parallel_map(process: Callable[[InT], OutT], items: List[InT], parallelism: int) -> List[OutT]:
    """
    A wrapper for applying `process` to all `items`.
    """
    with htrack_block(f"Parallelizing computation on {len(items)} items over {parallelism} threads"):
        results: List
        if parallelism == 1:
            results = list(tqdm(map(process, items), total=len(items), disable=None))
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


def unique_simplification(items: List[Dict[str, Any]], priority_keys: List[str]) -> List[Dict[str, Any]]:
    """
    Given `items` (a list of dictionaries), remove any (key, value) pairs that
    aren't necessary to distinguish the items, removing the keys not in
    `priority_keys` and then from the end of `priority_keys` first.

    Example:
        items = [{"model": "M1", stop: "#", n: 3}, {"model": "M1", stop: "\n", n: 3}, {"model": "M2", stop: "\n", n: 3}]
        priority_keys = ["model"]
    Return:
        [{"model": "M1", stop: "#"}, {"model": "M1", stop: "\n"}, {"model": "M2"}]
    """

    def get_subitem(item: Dict[str, Any], subkeys: List[str]) -> Dict[str, Any]:
        return {key: item.get(key) for key in subkeys}

    def get_keys(item: Dict[str, Any]) -> List[str]:
        """Return the keys of `item`, putting `priority_keys` first."""
        keys = []
        for key in priority_keys:
            if key in item:
                keys.append(key)
        for key in item:
            if key not in priority_keys:
                keys.append(key)
        return keys

    # Strip out common entries first
    items = without_common_entries(items)

    # Go through and remove more keys
    new_items: List[Dict[str, Any]] = []
    for item in items:
        # For each item, go through the keys in order
        keys = get_keys(item)

        for i in range(len(keys)):
            # For each prefix of the keys, keep it if it uniquely identifies
            # this item.
            subkeys = keys[: i + 1]
            subitem = get_subitem(item, subkeys)
            if sum(1 if get_subitem(item2, subkeys) == subitem else 0 for item2 in items) == 1:
                item = subitem
                break
        new_items.append(item)

    return new_items


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


def safe_symlink(src: str, dest: str) -> None:
    """
    Creates a symlink at `dest`. `src` and `dest` can be relative paths.
    """
    src = os.path.abspath(src)
    dest = os.path.abspath(dest)

    if not os.path.exists(dest):
        os.symlink(src, dest)


def is_url(location: str) -> bool:
    """Return True if `location` is a url. False otherwise."""
    return urllib.parse.urlparse(location).scheme in ["http", "https"]


def assert_is_str(val: Any) -> str:
    assert isinstance(val, str)
    return val


def assert_is_str_list(val: Any) -> List[str]:
    assert isinstance(val, list)
    for v in val:
        assert isinstance(v, str)
    return val


def assert_present(val: Optional[InT]) -> InT:
    assert val is not None
    return val
