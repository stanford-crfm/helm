import os
import json
import csv
import sys
from typing import List
from common.general import ensure_file_downloaded
from common.hierarchical_logger import hlog
from .scenario import Scenario, Instance, TEST_TAG

csv.field_size_limit(sys.maxsize)


class ThePileScenario(Scenario):
    """
    The The Pile corpus from this paper:

        https://arxiv.org/pdf/2101.00027.pdf
    """

    name = "the_pile"
    description = "The Pile"
    tags = ["language_modeling"]

    def __init__(self, subset: str):
        self.pile_subsets = {
            "ArXiv",
            "BookCorpus2",
            "Books3",
            "DM Mathematics",
            "Enron Emails",
            "EuroParl",
            "FreeLaw",
            "Github",
            "Gutenberg (PG-19)",
            "HackerNews",
            "NIH ExPorter",
            "OpenSubtitles",
            "OpenWebText2",
            "PhilPapers",
            "Pile-CC",
            "PubMed Abstracts",
            "PubMed Central",
            "StackExchange",
            "USPTO Backgrounds",
            "Ubuntu IRC",
            "Wikipedia (en)",
            "YoutubeSubtitles",
        }
        assert subset in self.pile_subsets
        self.subset = subset

    def load_and_cache_all_subsets(self):
        data_path = os.path.join(self.output_path, "data")
        hlog(f"Extracting subsets from {data_path}")
        subsets = {subset: [] for subset in self.pile_subsets}

        # Load all data into memory
        with open(data_path) as f:
            data = [json.loads(line) for line in f]

        # Classify the documents by subset
        for doc in data:
            subsets[doc["meta"]["pile_set_name"]].append([doc["text"]])

        # Write each subset to disk
        hlog(f"Caching subsets to {self.output_path}")
        for subset in subsets:
            subset_path = os.path.join(self.output_path, subset + ".csv")
            with open(subset_path, "w") as f:
                writer = csv.writer(f)
                writer.writerows(subsets[subset])

    def get_instances(self) -> List[Instance]:
        # Download the raw data
        data_path = os.path.join(self.output_path, "data")
        ensure_file_downloaded(
            source_url="https://mystic.the-eye.eu/public/AI/pile/test.jsonl.zst", target_path=data_path, unpack=True,
        )

        subset_path = os.path.join(self.output_path, self.subset + ".csv")

        # If the target subset does not exist, load and cache all subsets to the directory
        if not os.path.exists(subset_path):
            self.load_and_cache_all_subsets()

        # Read all the instances
        instances = []
        hlog(f"Reading {subset_path}")
        with open(subset_path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                instance = Instance(input=row[0], references=[], tags=[TEST_TAG],)
                instances.append(instance)

        return instances
