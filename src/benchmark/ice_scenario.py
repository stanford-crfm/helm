import os
import json
import csv
import sys
import requests
from typing import List
from common.general import ensure_file_downloaded
from common.hierarchical_logger import hlog, htrack, htrack_block
from .scenario import Scenario, Instance, TEST_SPLIT


class ICEScenario(Scenario):
    """
    The International Corpus of English (ICE).

    Documentation: https://www.ice-corpora.uzh.ch/en.html
    """

    name = "ice"
    description = "International Corpus of English (ICE)"
    tags = ["harms", "fairness", "language_modeling"]

    def init(self, subset: str, gender: str = None):
        self.ice_subsets = {"CAN", "GB", "JA", "HK", "EA", "IND", "SIN", "PHI", "USA", "IRL", "NZ", "SL", "NG"}
        assert subset in self.ice_subsets
        self.subset = subset

    @htrack(None)
    def load_and_cache_all_subsets(self):
        pass

    def get_instances(self) -> List[Instance]:
        # Download the raw data
        data_path = os.path.join(self.output_path, "data")
        ensure_file_downloaded(
            source_url=None, target_path=data_path, unpack=True,
        )
        return []
