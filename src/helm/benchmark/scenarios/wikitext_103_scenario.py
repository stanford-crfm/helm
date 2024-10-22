import os
import re
from typing import List

from helm.common.general import ensure_file_downloaded
from helm.common.hierarchical_logger import hlog
from helm.benchmark.scenarios.scenario import Scenario, Instance, TEST_SPLIT, Input


class Wikitext103Scenario(Scenario):
    """
    Wikitext-103 dataset from this paper:
    https://arxiv.org/pdf/1609.07843.pdf

    Gopher's authors concatenate all the articles, set context length to n/2 (n = max_seq_len),
    and use the "closed vocabulary" variant of the dataset for evaluation.

    In contrast, we evaluate the model on each article independently, use single token contexts
    (except for the last sequence in each document), and use the raw dataset.
    """

    name = "wikitext_103"
    description = "The WikiText language modeling dataset containing over 103 million words"
    tags = ["language_modeling"]

    def get_instances(self, output_path: str) -> List[Instance]:
        # Download the raw data
        data_path = os.path.join(output_path, "data")
        ensure_file_downloaded(
            source_url="https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip",
            target_path=data_path,
            unpack=True,
        )

        test_set_path = os.path.join(data_path, "wiki.test.raw")
        # regex patterns used for document extraction
        DOCUMENT_TITLE_PATTERN = "^ = .+ = $"
        PARAGRAPH_TITLE_PATTERN = "^ = = .+ = = $"
        EMPTY_LINE = " " + os.linesep

        # Read all the instances
        instances = []
        hlog(f"Reading {test_set_path}")
        lines = open(test_set_path, "r").readlines()
        document_buffer: List[str] = []
        i = 0
        # Example (some sections are empty):
        #
        # = Article Title =
        #
        # Some introduction to the topic.
        #
        # = = Section 1 = =
        #
        #
        # = = = Subsection 1.1 = = =
        #
        # Content of Subsection 1.1.
        #
        # = = Section 2 = =
        #
        #
        while i < len(lines):
            if (
                i <= len(lines) - 3
                and lines[i] == EMPTY_LINE
                and lines[i + 2] == EMPTY_LINE
                and re.match(DOCUMENT_TITLE_PATTERN, lines[i + 1])
                and not re.match(PARAGRAPH_TITLE_PATTERN, lines[i + 1])
            ):
                # Create an instance and append it to instances
                if document_buffer:
                    instance = Instance(Input(text="".join(document_buffer)), references=[], split=TEST_SPLIT)
                    instances.append(instance)
                document_buffer = lines[i : i + 3]
                i += 3
            else:
                document_buffer.append(lines[i])
                i += 1
        # Add the last instance to instances
        if document_buffer:
            instance = Instance(Input(text="".join(document_buffer)), references=[], split=TEST_SPLIT)
            instances.append(instance)

        # The test set of Wikitext-103 contains 60 articles
        assert len(instances) == 60
        return instances
