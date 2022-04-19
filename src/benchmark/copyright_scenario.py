import json
import os
from typing import List

import tqdm

from common.general import ensure_file_downloaded
from .scenario import Scenario, Instance, Reference, CORRECT_TAG, TEST_SPLIT

datatag2hash = {
    # Very small; 10 examples.
    "pilot": "https://drive.google.com/file/d/1NwzDx19uzIwBuw7Lq5CSytG7jIth2wJ-/view?usp=sharing",
    # 1k examples.
    "n_books_1000-extractions_per_book_1-prefix_length_5": "https://drive.google.com/file/d/1_B8xfXQTklaAXgOfeSCuL3FvXvoXGBPq/view?usp=sharing",
    "n_books_1000-extractions_per_book_1-prefix_length_25": "https://drive.google.com/file/d/1i-v-KACEUnKOljJxfe5u_qcrA-uTBCky/view?usp=sharing",
    "n_books_1000-extractions_per_book_1-prefix_length_125": "https://drive.google.com/file/d/1TRbkha807PiDKoegA6Kqf9SgqBUOlM1Y/view?usp=sharing",
    # 3k examples from 1k books.
    "n_books_1000-extractions_per_book_3-prefix_length_5": "https://drive.google.com/file/d/1tXfhaAZrwkA4C7TnqU7SV4gj2RxIIT4x/view?usp=sharing",
    "n_books_1000-extractions_per_book_3-prefix_length_25": "https://drive.google.com/file/d/1ciKUFfCh1MKQ1m6KFEHdzX4AWy_ec0R9/view?usp=sharing",
    "n_books_1000-extractions_per_book_3-prefix_length_125": "https://drive.google.com/file/d/1MZyMoCSgA6-_hu4gQe4G9IfeyVqgwfUb/view?usp=sharing",
}


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

    def __init__(self, datatag="pilot"):
        self.datatag = datatag
        self.source_url = datatag2hash[datatag]

    def get_instances(self) -> List[Instance]:
        data_path = os.path.join(self.output_path, "data")
        ensure_file_downloaded(
            source_url=self.source_url, target_path=data_path,
        )
        with open(data_path, "r") as f:
            # Processed data with the format {"data": {prefix: prefix_to_end}, "metadata":...}.
            data = json.load(f)

        # Read all the instances
        instances = []
        for prefix, prefix_to_end in tqdm.tqdm(data["data"].items(), desc="load instances"):
            instances.append(
                Instance(
                    input=prefix, references=[Reference(output=prefix_to_end, tags=[CORRECT_TAG])], split=TEST_SPLIT,
                ),
            )
        return instances
