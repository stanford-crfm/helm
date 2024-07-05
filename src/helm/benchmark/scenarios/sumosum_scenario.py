import os
from datasets import load_dataset
import pandas as pd
import csv

from typing import List, Optional
from helm.common.general import ensure_file_downloaded, ensure_directory_exists
from .scenario import Scenario, Instance, Reference, TRAIN_SPLIT, TEST_SPLIT, CORRECT_TAG, Input, Output


class SUMOSumScenario(Scenario):
    """
    Generating Fact Checking Summaries for Web Claims ([paper](https://arxiv.org/abs/2010.08570))
    """

    TRAIN_RATIO: float = 0.2

    name = "sumosum"
    description = "Text summarization with climate corpus"
    tags = ["summarization", "climate"]

    def __init__(
        self,
        sampling_min_length: Optional[int] = None,
        sampling_max_length: Optional[int] = None,
        doc_max_length: Optional[int] = None,
    ):
        """
        Initializes the scenario.
        Args:
            sampling_min_length: Int indicating minimum length for training
                                 documents. Training examples smaller than
                                 sampling_min_length will be filtered out.
                                 Useful for preventing the adapter from sampling
                                 really small documents.
            sampling_max_length: Int indicating maximum length for training
                                 documents. Training examples larger than
                                 sampling_max_length will be filtered out.
                                 Useful for preventing the adapter from
                                 sampling really large documents.
            doc_max_length: Int indicating the maximum length to truncate
                            documents. Documents in all splits will be
                            truncated to doc_max_length tokens.
                            NOTE: Currently uses whitespace tokenization.
        """
        super().__init__()
        self.sampling_min_length = sampling_min_length
        self.sampling_max_length = sampling_max_length
        self.doc_max_length = doc_max_length

    @staticmethod
    def _clean_and_truncate(text: str, max_length: Optional[int] = None) -> str:
        text = text.replace("\n", " ")
        return " ".join(text.split()[:max_length])

    def _load_dataset(self, output_path: str):
        data_dir = os.path.join(output_path, "data")
        ensure_directory_exists(data_dir)

        source_url = "https://github.com/rahulOmishra/SUMO/raw/main/climate_claims_raw.xlsx"
        source_file = os.path.basename(source_url)
        target_path = os.path.join(data_dir, source_file)
        ensure_file_downloaded(
            source_url=source_url,
            target_path=target_path,
        )

        source_file_noext = os.path.splitext(source_file)[0]
        train_file = f"{source_file_noext}-{TRAIN_SPLIT}.csv"
        test_file = f"{source_file_noext}-{TEST_SPLIT}.csv"
        title_key = "Title"
        document_key = "Doc_text"

        # Claim_id(int),Claim,Title,Doc_text,Label(bool)
        target_df = pd.read_excel(target_path, skiprows=1)
        target_df = target_df.dropna(subset=[title_key, document_key])
        # Remove carriage return _x000D_ in Excel string
        target_df = target_df.replace({r"_x000D_": ""}, regex=True)
        # target_df = target_df.replace({r"_x([0-9a-fA-F]{4})_": ""}, regex=True)
        # Split randomly (works better than split by order)
        train_df = target_df.sample(frac=SUMOSumScenario.TRAIN_RATIO, random_state=0)
        test_df = target_df.drop(train_df.index).sample(frac=1, random_state=0)
        train_df.to_csv(os.path.join(data_dir, train_file), index=False, header=True, quoting=csv.QUOTE_NONNUMERIC)
        test_df.to_csv(os.path.join(data_dir, test_file), index=False, header=True, quoting=csv.QUOTE_NONNUMERIC)

        data_files = {}
        data_files[TRAIN_SPLIT] = train_file
        data_files[TEST_SPLIT] = test_file
        dataset = load_dataset(data_dir, data_files=data_files)

        return dataset, title_key, document_key

    def get_instances(self, output_path: str) -> List[Instance]:
        dataset, title_key, document_key = self._load_dataset(output_path)

        instances: List[Instance] = []

        SKIP_MAX_LENGTH = 3700  # A too-long article doesn't fit in a prompt.
        SKIP_MIN_LENGTH = 100  # A too-short article could be garbage.

        for split, split_data in dataset.items():
            for example in split_data:
                document: str = SUMOSumScenario._clean_and_truncate(example[document_key])
                # NOTE Select relatively short documents and truncate them to preserve as much information from the original documents as possible  # noqa
                if split in TEST_SPLIT:
                    art_len = len(document.split())
                    if art_len > SKIP_MAX_LENGTH:
                        continue
                    if art_len < SKIP_MIN_LENGTH:
                        continue
                document = SUMOSumScenario._clean_and_truncate(example[document_key], self.doc_max_length)

                title: str = SUMOSumScenario._clean_and_truncate(example[title_key])

                if split == TRAIN_SPLIT:
                    art_len = len(document.split())
                    if self.sampling_max_length and art_len > self.sampling_max_length:
                        continue
                    if self.sampling_min_length and art_len < self.sampling_min_length:
                        continue

                instance = Instance(
                    input=Input(text=document),
                    references=[Reference(output=Output(text=title), tags=[CORRECT_TAG])],
                    split=split,
                )
                instances.append(instance)

        return instances
