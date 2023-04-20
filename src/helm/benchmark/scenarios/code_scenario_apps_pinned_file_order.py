import json
import os
from typing import Dict, List

from helm.common.general import ensure_file_downloaded


_APPS_PINNED_FILE_ORDER_URL: str = (
    "https://worksheets.codalab.org/rest/bundles/0x308e1062e2b7448bacc45b4c92af5a75/contents/blob/"
)
_APPS_PINNED_FILE_ORDER_FILENAME: str = "code_scenario_apps_pinned_file_order.json"


_apps_split_to_pinned_file_order: Dict[str, List[str]] = {}
"""Dict of subdirectory to pinned file order."""


def get_apps_split_to_pinned_file_order(download_dir: str):
    """Lazily download and return a dict of subdirectory to pinned file order."""
    global _apps_split_to_pinned_file_order
    if not _apps_split_to_pinned_file_order:
        file_path: str = os.path.join(download_dir, _APPS_PINNED_FILE_ORDER_FILENAME)
        ensure_file_downloaded(
            source_url=_APPS_PINNED_FILE_ORDER_URL,
            target_path=file_path,
            unpack=False,
        )
        with open(file_path) as f:
            _apps_split_to_pinned_file_order = json.load(f)
    return _apps_split_to_pinned_file_order


def apps_listdir_with_pinned_order(download_dir: str, split_tag: str) -> List[str]:
    """List files for the split in a pinned order for APPS to ensure reproducibility.

    Unfortunately, the previous official HELM runs used the arbitrary file order
    produced by os.listdir() and did not sort the files, so future runs must use
    the same file order in order to sample the same test instances to reproduce
    the official HELM runs."""
    pinned_filename_order = get_apps_split_to_pinned_file_order(download_dir).get(split_tag)
    if pinned_filename_order:
        return pinned_filename_order
    return list(sorted(os.listdir(os.path.join(download_dir, split_tag))))
