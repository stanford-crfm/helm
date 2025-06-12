import json
import os
import tqdm
from typing import List

from helm.common.general import ensure_file_downloaded
from helm.benchmark.scenarios.scenario import Scenario, Instance, Reference, CORRECT_TAG, TEST_SPLIT, Input, Output

datatag2hash_text = {
    # The "average" book.
    # Very small; 10 examples.
    "pilot": "1NwzDx19uzIwBuw7Lq5CSytG7jIth2wJ-",
    # 1k examples.
    "n_books_1000-extractions_per_book_1-prefix_length_5": "16nQD8Nq3ma4K2EZLXHcahG9fcBDxS-zB",
    "n_books_1000-extractions_per_book_1-prefix_length_25": "108sZcMjzY7mvyy1p5Rw62_A8A1I2zM2S",
    "n_books_1000-extractions_per_book_1-prefix_length_125": "10uC4jM6tgI1pgtq--07FFHQ2Te7-SXGA",
    # 3k examples from 1k books.
    "n_books_1000-extractions_per_book_3-prefix_length_5": "1byrafXv2iULcZArxguJZp2LyFxswX7fN",
    "n_books_1000-extractions_per_book_3-prefix_length_25": "13QOKOd5Fpu5cVu1HRBYxzRcQzwhcPhjD",
    "n_books_1000-extractions_per_book_3-prefix_length_125": "1Y6QvYStCJVanHaI67Pxep3HakWL2cIRP",
    # 20 popular books.
    "popular_books-prefix_length_5.json": "11f5aB8IO_iseVdolvW_D3aZ9xeiLCAuv",
    "popular_books-prefix_length_10.json": "1oG9dMz1WiJZiuoXqw4ahFf1HmKWQtwJh",
    "popular_books-prefix_length_25.json": "1dN208HdBD6koaOHQlSDPkOnBrY3-9AfE",
    "popular_books-prefix_length_50.json": "1W_8C6QnISfFkc0tZtKdXKNhoTfgSCrbP",
    "popular_books-prefix_length_125.json": "1RT29rRKNNXKgZBhXNbqevLwR440g44it",
    "popular_books-prefix_length_250.json": "1KcQ3EJGAZO6fSqYXRH5eK122h1hUKZTp",
    # "Oh, the Places You'll Go!" by Dr. Seuss. Only contains 3 prompts; demo for the followup paper.
    "oh_the_places": "1KXqhO14HmCGQ67tuu2ECGbTGloTkBvzt",
}
datatag2hash_code = {
    # Linux kernel source code.
    "prompt_num_line_1-min_lines_20.json": "1OLFyW5u7govgIw3ztsZ_5yYV0YpGzi-3",
    "prompt_num_line_5-min_lines_20.json": "1YbDvyAv9hT0BaZ5LV6Y-Y8tGezrBnBAT",
    "prompt_num_line_10-min_lines_20.json": "1Y5piYwil7T6n8toT_-d7NWqVZHh9NVxJ",
}
datatag2hash = {**datatag2hash_text, **datatag2hash_code}


class CopyrightScenario(Scenario):
    """Test the risk of disqualifying for fair use via data extraction attack.

    Each instance in this scenario contains

    1. a randomly sampled prefix from the bookcorpus, and
    2. the entire remaining book.

    Methodology adapted from
        Carlini, Nicholas, et al.
        "Extracting training data from large language models."
        30th USENIX Security Symposium (USENIX Security 21). 2021.
    """

    name = "copyright"
    description = (
        "Data extraction attacks based on the original BookCorpus, curated popular books, and GPL source code."
    )
    tags = ["harms", "copyright"]

    def __init__(self, datatag="pilot"):
        super().__init__()
        self.datatag = datatag
        self.source_url = f"https://drive.google.com/uc?id={datatag2hash[datatag]}"

    def get_instances(self, output_path: str) -> List[Instance]:
        target_path: str = os.path.join(output_path, f"{self.datatag}.json")
        ensure_file_downloaded(self.source_url, target_path)

        with open(target_path, "r") as f:
            # Processed data with the format {"data": {prefix: prefix_to_end}, "metadata":...}.
            data = json.load(f)

        # Read all the instances
        instances: List[Instance] = []
        for prefix, prefix_to_end in tqdm.tqdm(data["data"].items(), desc="load instances", disable=None):
            instances.append(
                Instance(
                    input=Input(text=prefix),
                    references=[Reference(Output(text=prefix_to_end), tags=[CORRECT_TAG])],
                    split=TEST_SPLIT,
                ),
            )
        return instances
