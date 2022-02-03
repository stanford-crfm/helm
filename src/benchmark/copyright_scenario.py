import json
import os
from typing import List

import tqdm

from common.general import ensure_file_downloaded
from .scenario import Scenario, Instance, Reference, CORRECT_TAG, TEST_TAG

PILOT_DATA_URL = "https://docs.google.com/uc?export=download&id=1NwzDx19uzIwBuw7Lq5CSytG7jIth2wJ-"


class CopyrightScenario(Scenario):
    """Test the risk of disqualifying for fair use via data extraction attack.

    Each instance in this scenario contains
        1) a randomly sampled prefix from the bookcorpus, and
        2) the entire remaining book.

    Methodology adapted from
        Carlini, Nicholas, et al.
        "Extracting training data from large language models."
        30th USENIX Security Symposium (USENIX Security 21). 2021.
    """

    name = "copyright"
    description = "Data extraction attacks based on the BookCorpus."
    tags = ["harms", "copyright"]

    def __init__(self):
        pass

    def get_instances(self) -> List[Instance]:
        # Download the raw data
        data_path = os.path.join(self.output_path, "data")
        ensure_file_downloaded(
            source_url=PILOT_DATA_URL,  # TODO: Add in full data after successful processing.
            target_path=data_path,
        )
        with open(data_path, 'r') as f:
            # Processed data with the format {"data": {prefix: completion}, "metadata":...}.
            data = json.load(f)

        # Read all the instances
        instances = []
        for prefix, completion in tqdm.tqdm(data["data"].items(), desc='load instances'):
            completion_minus_prefix = completion[len(prefix):]
            instances.append(
                Instance(
                    input=prefix,
                    references=[Reference(output=completion_minus_prefix, tags=[CORRECT_TAG])],
                    tags=[TEST_TAG],  # Must assign split tag to instance.
                ),
            )
        return instances
