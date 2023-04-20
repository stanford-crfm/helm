import os
import json
from typing import Dict, List

from helm.common.general import ensure_file_downloaded


_PINNED_FILE_ORDER_URL: str = (
    "https://worksheets.codalab.org/rest/bundles/0xdee5cea197f645219093954ac7324add/contents/blob/"
)
_PINNED_FILE_ORDER_FILENAME: str = "ice_scenario_pinned_file_order.json"


_path_to_pinned_file_order: Dict[str, List[str]] = {}
"""Dict of path to pinned file order."""


def get_path_to_pinned_file_order(download_dir: str):
    """Lazily download and return a dict of path to pinned file order."""
    global _path_to_pinned_file_order
    if not _path_to_pinned_file_order:
        file_path: str = os.path.join(download_dir, _PINNED_FILE_ORDER_FILENAME)
        ensure_file_downloaded(
            source_url=_PINNED_FILE_ORDER_URL,
            target_path=file_path,
            unpack=False,
        )
        with open(file_path) as f:
            _path_to_pinned_file_order = json.load(f)
    return _path_to_pinned_file_order


def listdir_with_pinned_file_order(download_dir: str, corpus_path: str) -> List[str]:
    """List files for the path in a pinned order for ICE to ensure reproducibility.

    Unfortunately, the previous official HELM runs used the arbitrary file order
    produced by os.listdir() and did not sort the files, so future runs must use
    the same file order in order to sample the same instances to reproduce
    the official HELM runs."""
    pinned_file_order = get_path_to_pinned_file_order(download_dir).get(corpus_path)
    if pinned_file_order:
        return pinned_file_order
    return list(sorted(os.listdir(corpus_path)))
