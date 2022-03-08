import os
import re

from typing import List
from common.general import ensure_file_downloaded
from common.hierarchical_logger import hlog
from .scenario import Scenario, Instance, TEST_SPLIT


class Wikitext103Scenario(Scenario):
    """
    Wikitext-103 dataset from this paper:

        https://arxiv.org/pdf/1609.07843.pdf
    """

    name = "wikitext_103"
    description = "Wikitext-103"
    tags = ["language_modeling"]

    def __init__(self):
        pass

    def get_instances(self) -> List[Instance]:
        # Download the raw data
        data_path = os.path.join(self.output_path, "data")
        ensure_file_downloaded(
            source_url="https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip",
            target_path=data_path,
            unpack=True,
        )

        test_set_path = os.path.join(data_path, "wiki.test.raw")
        # regex patterns used for document extraction
        document_title_pattern = "^ = .+ = $"
        paragraph_title_pattern = "^ = = .+ = = $"

        # Read all the instances
        instances = []
        hlog(f"Reading {test_set_path}")
        with open(test_set_path, "r") as f:
            # Following the raw data format, prepend a line seperator to every document
            document_buffer = [os.linesep]
            for line in f:
                if re.match(document_title_pattern, line) and not re.match(paragraph_title_pattern, line):
                    # Create an instance and append it to instances
                    instance = Instance(input="".join(document_buffer[:-1]), references=[], split=TEST_SPLIT)
                    instances.append(instance)
                    # empty the buffer
                    document_buffer = [os.linesep]
                document_buffer.append(line)
        return instances
