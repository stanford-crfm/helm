import csv
import os
from typing import List

from helm.common.general import ensure_file_downloaded
from helm.common.hierarchical_logger import hlog
from helm.benchmark.scenarios.scenario import Scenario, Instance, TEST_SPLIT, Input

CODALAB_URI_TEMPLATE: str = (
    "https://worksheets.codalab.org/rest/bundles/0x31485f8c37ad481fb9f4e9bf7ccff6e5/contents/blob/"
    "{demographic}_tweets.csv"
)


class TwitterAAEScenario(Scenario):
    """
    The TwitterAAE corpus from this paper:
    https://aclanthology.org/D16-1120.pdf

    Our AA and white datasets are different from the AA-aligned and white-aligned corpora in the paper.

    Specificaly, we derive the datasets in two steps:

    1. Select the 830,000 tweets with the highest AA proportions and 7.3 million tweets with the highest
    white proportions from the source dataset.
    2. Randomly sample 50,000 tweets from each demographic subset as our test set.
    """

    name = "twitter_aae"
    description = "Twitter African-American English"
    tags = ["bias", "language_modeling"]

    def __init__(self, demographic: str = "aa"):
        super().__init__()
        assert demographic in ["aa", "white"], f"Unsupported demographic: {demographic}"
        self.demographic: str = demographic

    def get_instances(self, output_path: str) -> List[Instance]:
        # Download the raw data
        data_path: str = os.path.join(output_path, f"{self.demographic}_tweets.csv")
        ensure_file_downloaded(
            source_url=CODALAB_URI_TEMPLATE.format(demographic=self.demographic),
            target_path=data_path,
            unpack=False,
        )

        # Read all the instances
        instances: List[Instance] = []
        hlog(f"Reading {data_path}")
        with open(data_path) as f:
            reader = csv.reader(f)
            for row in reader:
                # Example: ["Click Clack Motha Fucka I ain't tryin to hear Nothin!"]
                tweet: str = row[0]
                instance = Instance(Input(text=tweet), references=[], split=TEST_SPLIT)
                instances.append(instance)

        return instances
