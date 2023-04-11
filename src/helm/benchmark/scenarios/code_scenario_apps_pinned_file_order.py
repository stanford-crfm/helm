import importlib_resources as resources
import json
import os
from typing import Dict, List


_SCENARIOS_DATA_PACKAGE: str = "helm.benchmark.scenarios"
_APPS_PINNED_FILE_ORDER_FILENAME: str = "code_scenario_apps_pinned_file_order.json"


_apps_pinned_file_order: Dict[str, List[str]] = {}


def _get_apps_pinned_file_order():
    """Lazily load the saved pinned file order for APPS."""
    global _apps_pinned_file_order
    if not _apps_pinned_file_order:
        data_package = resources.files(_SCENARIOS_DATA_PACKAGE)
        with data_package.joinpath(_APPS_PINNED_FILE_ORDER_FILENAME).open("r") as f:
            _apps_pinned_file_order = json.load(f)
    return _apps_pinned_file_order


def apps_listdir_with_pinned_order(split_tag: str, split_dir: str) -> List[str]:
    """List files for the split in a pinned order for APPS to ensure reproducibility.

    Unfortunately, the previous official HELM runs used the arbitrary file order
    produced by os.listdir() and did not sort the files, so future runs must use
    the same file order in order to sample the same test instances to reproduce
    the official HELM runs."""
    pinned_filename_order = _get_apps_pinned_file_order().get(split_tag)
    if pinned_filename_order:
        return pinned_filename_order
    return list(sorted(os.listdir(split_dir)))
