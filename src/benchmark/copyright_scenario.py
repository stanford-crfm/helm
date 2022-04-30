import json
import os
import tqdm
from typing import List

from .scenario import Scenario, Instance, Reference, CORRECT_TAG, TEST_SPLIT

datatag2hash = {
    # Very small; 10 examples.
    "pilot": "1NwzDx19uzIwBuw7Lq5CSytG7jIth2wJ-",
    # 1k examples.
    "n_books_1000-extractions_per_book_1-prefix_length_5": "1_B8xfXQTklaAXgOfeSCuL3FvXvoXGBPq",
    "n_books_1000-extractions_per_book_1-prefix_length_25": "1i-v-KACEUnKOljJxfe5u_qcrA-uTBCky",
    "n_books_1000-extractions_per_book_1-prefix_length_125": "1TRbkha807PiDKoegA6Kqf9SgqBUOlM1Y",
    # 3k examples from 1k books.
    "n_books_1000-extractions_per_book_3-prefix_length_5": "1tXfhaAZrwkA4C7TnqU7SV4gj2RxIIT4x",
    "n_books_1000-extractions_per_book_3-prefix_length_25": "1ciKUFfCh1MKQ1m6KFEHdzX4AWy_ec0R9",
    "n_books_1000-extractions_per_book_3-prefix_length_125": "1MZyMoCSgA6-_hu4gQe4G9IfeyVqgwfUb",
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
        self.source_url = f"https://drive.google.com/uc?id={datatag2hash[datatag]}"

    def get_instances(self) -> List[Instance]:
        target_path = os.path.join(self.output_path, f"{self.datatag}.json")
        # `ensure_file_downloaded` in src.common doesn't work.
        # The main problem is that naive wget cannot bypass the gdrive large file virus scan warning.
        if not os.path.exists(target_path):
            os.system(f"gdown {self.source_url} -O {target_path}")
        with open(target_path, "r") as f:
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
