import json
import os
from typing import Dict, List

from helm.common.general import ensure_file_downloaded


_PINNED_FILE_ORDER_URL: str = (
    "https://worksheets.codalab.org/rest/bundles/0xa75fa6eb8dc24e23a73d3a5ecb674b32/contents/blob/"
)
_PINNED_FILE_ORDER_FILENAME: str = "imdb_scenario_pinned_file_order.json"


_split_to_class_to_pinned_file_order: Dict[str, List[str]] = {}
"""Nested mapping of split to class to to pinned file order."""


def get_split_to_class_to_pinned_file_order(download_dir: str):
    """Lazily download and return a nested mapping of split to class to pinned file order."""
    global _split_to_class_to_pinned_file_order
    if not _split_to_class_to_pinned_file_order:
        file_path: str = os.path.join(download_dir, _PINNED_FILE_ORDER_FILENAME)
        ensure_file_downloaded(
            source_url=_PINNED_FILE_ORDER_URL,
            target_path=file_path,
            unpack=False,
        )
        with open(file_path) as f:
            _split_to_class_to_pinned_file_order = json.load(f)
    return _split_to_class_to_pinned_file_order


def listdir_with_pinned_file_order(download_dir: str, split: str, class_name: str) -> List[str]:
    """List files for the split in a pinned order for IMDB to ensure reproducibility.

    Unfortunately, the previous official HELM runs used the arbitrary file order
    produced by os.listdir() and did not sort the files, so future runs must use
    the same file order in order to sample the same instances to reproduce
    the official HELM runs."""
    return get_split_to_class_to_pinned_file_order(download_dir)[split][class_name]
