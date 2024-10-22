import os
import json
import csv
import sys
import requests
from typing import Dict, List

from helm.common.general import ensure_file_downloaded
from helm.common.hierarchical_logger import hlog, htrack, htrack_block
from helm.benchmark.scenarios.scenario import Scenario, Instance, TEST_SPLIT, Input


class ThePileScenario(Scenario):
    """
    The Pile corpus from this paper:
    https://arxiv.org/pdf/2101.00027.pdf
    """

    name = "the_pile"
    description = "The Pile"
    tags = ["language_modeling"]

    def __init__(self, subset: str):
        super().__init__()
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

    @htrack(None)
    def load_and_cache_all_subsets(self, data_jsonl, output_path):
        subsets: Dict[str, List] = {subset: [] for subset in self.pile_subsets}

        # Load all data into memory
        with htrack_block("Loading"):
            hlog(f"Loading all data from {data_jsonl}")
            with open(data_jsonl) as f:
                data = [json.loads(line) for line in f]

        # Classify the documents by subset
        hlog("Classifying the documents into subsets")
        for doc in data:
            subsets[doc["meta"]["pile_set_name"]].append([doc["text"]])

        # Write each subset to disk
        with htrack_block("Caching"):
            hlog(f"Caching subsets to {output_path}")
            for subset in subsets:
                subset_path = os.path.join(output_path, subset + ".csv")
                with open(subset_path, "w") as f:
                    writer = csv.writer(f)
                    writer.writerows(subsets[subset])

    def get_instances(self, output_path: str) -> List[Instance]:
        # Download the raw data
        data_jsonl = os.path.join(output_path, "data")
        ensure_file_downloaded(
            source_url="https://the-eye.eu/public/AI/pile/test.jsonl.zst",
            target_path=data_jsonl,
            unpack=True,
        )

        subset_path = os.path.join(output_path, self.subset + ".csv")

        # If the target subset does not exist, load and cache all subsets to the directory
        if not os.path.exists(subset_path):
            self.load_and_cache_all_subsets(data_jsonl, output_path)

        # Read all the instances
        instances = []
        hlog(f"Reading {subset_path}")
        with open(subset_path, "r") as f:
            csv.field_size_limit(sys.maxsize)
            reader = csv.reader(f)
            for row in reader:
                instance = Instance(
                    input=Input(text=row[0]),
                    references=[],
                    split=TEST_SPLIT,
                )
                instances.append(instance)

        # Load the subsample indices
        # Short names of the datasets
        DATASET_NAMES_DICT = {
            "Github": "github",
            "ArXiv": "arxiv",
            "Wikipedia (en)": "wikipedia",
            "OpenSubtitles": "opensubtitles",
            "OpenWebText2": "openwebtext2",
            "Gutenberg (PG-19)": "gutenberg",
            "DM Mathematics": "dm-mathematics",
            "Enron Emails": "enron",
            "Books3": "bibliotik",
            "PubMed Abstracts": "pubmed-abstracts",
            "YoutubeSubtitles": "youtubesubtitles",
            "HackerNews": "hackernews",
            "Pile-CC": "commoncrawl",
            "EuroParl": "europarl",
            "USPTO Backgrounds": "uspto",
            "FreeLaw": "freelaw",
            "NIH ExPorter": "nih-exporter",
            "StackExchange": "stackexchange",
            "PubMed Central": "pubmed-central",
            "Ubuntu IRC": "ubuntu-irc",
            "BookCorpus2": "bookcorpus",
            "PhilPapers": "philpapers",
        }

        # These datasets were too small (in number of docs) to split 10-ways
        DATASETS_WITHOUT_SPLIT = [
            "ubuntu-irc",
            "bookcorpus",
            "philpapers",
        ]

        short_name = DATASET_NAMES_DICT[self.subset]
        if short_name not in DATASETS_WITHOUT_SPLIT:
            url = (
                "https://raw.githubusercontent.com/EleutherAI/lm_perplexity/main/assets/test_subsample_indices/"
                f"{short_name}/group0.json"
            )
            indices = sorted(list(set(requests.get(url).json())))
            instances = [instances[i] for i in indices]

        return instances
