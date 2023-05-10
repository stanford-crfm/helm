import json
import os
from typing import Dict, Tuple

import numpy as np

from helm.common.general import ensure_file_downloaded


RandomState = Tuple[str, np.ndarray, int, int, float]


_FIXED_RANDOM_SEED_URL: str = (
    "https://worksheets.codalab.org/rest/bundles/0x4d0a972e99c848d2ab3ecf129bb77d29/contents/blob/"
)
_FIXED_RANDOM_SEED_FILENAME: str = "entity_matching_scenario_fixed_random_state.json"


_split_to_fixed_random_seed: Dict[str, RandomState] = {}
"""Dict of dataset to fixed random seed."""


def get_split_to_fixed_random_seed(download_dir: str):
    """Lazily download and return a dict of dataset to fixed random seed."""
    global _split_to_fixed_random_seed
    if not _split_to_fixed_random_seed:
        file_path: str = os.path.join(download_dir, _FIXED_RANDOM_SEED_FILENAME)
        ensure_file_downloaded(
            source_url=_FIXED_RANDOM_SEED_URL,
            target_path=file_path,
            unpack=False,
        )
        with open(file_path) as f:
            raw_split_to_fixed_random_seed = json.load(f)
            for dataset, raw_fixed_random_seed in raw_split_to_fixed_random_seed.items():
                _split_to_fixed_random_seed[dataset] = (
                    str(raw_fixed_random_seed[0]),
                    np.array(
                        raw_fixed_random_seed[1],
                        dtype=np.uint32,
                    ),
                    int(raw_fixed_random_seed[2]),
                    int(raw_fixed_random_seed[3]),
                    int(raw_fixed_random_seed[4]),
                )
    return _split_to_fixed_random_seed


def set_fixed_random_state_for_dataset(download_dir: str, dataset: str) -> None:
    """Set the fixed random state for entity_matching_scenario to ensure reproducibility.

    Unfortunately, the previous official HELM runs did not initialize
    the numpy random state to zero, so future runs must use the same
    random states in order to sample the same test instances to reproduce
    the official HELM runs."""
    random_state = get_split_to_fixed_random_seed(download_dir).get(dataset)
    if random_state:
        np.random.set_state(random_state)
    else:
        np.random.seed(0)
