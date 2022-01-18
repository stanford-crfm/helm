import csv
import os
from typing import List
from common.general import ensure_file_downloaded
from common.hierarchical_logger import hlog
from .scenario import Scenario, Instance, Reference, TRAIN_TAG, VALID_TAG, TEST_TAG, CORRECT_TAG


class TwitterAAEScenario(Scenario):
    """
    The TwitterAAE corpus from this paper:

        https://aclanthology.org/D16-1120.pdf

    Code is adapted from:

        https://github.com/hendrycks/test/blob/master/evaluate.py
        https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/hendrycks_test.py
    """

    name = "twitter_aae"
    description = "Twitter African-American English"
    tags = ["bias", "language_modeling"]

    def __init__(self, demographic: str = "aa"):
        self.demographics = {"aa": 0, "white": 1}
        assert demographic in self.demographics
        self.demographic = demographic

    def get_instances(self) -> List[Instance]:
        # Download the raw data
        data_path = os.path.join(self.output_path, "data")
        ensure_file_downloaded(
            source_url="http://slanglab.cs.umass.edu/TwitterAAE/TwitterAAE-full-v1.zip",
            target_path=data_path,
            unzip=True,
        )

        # Read all the instances
        instances = []

        # TODO: This is a fake dataset just for testing purpose!
        tsv_path = os.path.join(data_path, "twitteraae_all_aa")
        if not os.path.exists(tsv_path):
            raise Exception(f"{tsv_path} doesn't exist")
        hlog(f"Reading {tsv_path}")
        with open(tsv_path) as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                # Example: ['293846693215096832', 'Tue Jan 22 22:24:45 +0000 2013', '1028920752', '[-80.01040975, 32.80108357]', '450190027021', "Click Clack Motha Fucka I ain't tryin to hear Nothin!", '0.894545454545', '0.0163636363636', '0.0', '0.0890909090909']
                tweet = row[5]
                instance = Instance(input=tweet, references=[], tags=[TEST_TAG],)
                instances.append(instance)

        return instances
