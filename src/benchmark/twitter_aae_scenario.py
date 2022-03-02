import csv
import os
from typing import List
from common.general import ensure_file_downloaded
from common.hierarchical_logger import hlog
from .scenario import Scenario, Instance, TEST_SPLIT


class TwitterAAEScenario(Scenario):
    """
    The TwitterAAE corpus from this paper:

        https://aclanthology.org/D16-1120.pdf

    The current implementation of this scenario is incomplete and should only be used for testing purposes.
    It does not support `demographic=white`; the `aa` subset is unfiltered, either.
    # TODO: Implement the data filtering process for `white` and `aa`
    #       https://github.com/stanford-crfm/benchmarking/issues/73
    """

    name = "twitter_aae"
    description = "Twitter African-American English"
    tags = ["bias", "language_modeling"]

    def __init__(self, demographic: str = "aa"):
        assert demographic in ["aa", "white"], f"Unsupported demographic: {demographic}"
        self.demographic: str = demographic

    def get_instances(self) -> List[Instance]:
        # Download the raw data
        data_path: str = os.path.join(self.output_path, "data")
        ensure_file_downloaded(
            source_url="http://slanglab.cs.umass.edu/TwitterAAE/TwitterAAE-full-v1.zip",
            target_path=data_path,
            unpack=True,
            unpack_type="unzip",
        )

        # Read all the instances
        instances: List[Instance] = []

        tsv_path: str = os.path.join(data_path, "twitteraae_all_aa")
        if not os.path.exists(tsv_path):
            raise Exception(f"{tsv_path} doesn't exist")

        hlog(f"Reading {tsv_path}")
        with open(tsv_path) as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                # Example: ['293846693215096832', 'Tue Jan 22 22:24:45 +0000 2013', '1028920752',
                # '[-80.01040975, 32.80108357]', '450190027021',"Click Clack Motha Fucka I ain't tryin to hear Nothin!",
                # '0.894545454545', '0.0163636363636', '0.0', '0.0890909090909']
                tweet = row[5]
                instance = Instance(input=tweet, references=[], split=TEST_SPLIT)
                instances.append(instance)

        return instances
