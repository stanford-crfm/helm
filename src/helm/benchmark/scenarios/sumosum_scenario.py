import csv
import os
import re
from typing import Dict, List, Mapping, Optional

from datasets import Dataset, load_dataset
import pandas as pd

from helm.common.general import ensure_file_downloaded, ensure_directory_exists
from helm.benchmark.scenarios.scenario import Scenario, Instance, Reference, TRAIN_SPLIT, TEST_SPLIT, CORRECT_TAG, Input, Output
from helm.common.optional_dependencies import handle_module_not_found_error


class SUMOSumScenario(Scenario):
    """
    Generating Fact Checking Summaries for Web Claims ([paper](https://arxiv.org/abs/2010.08570))
    """

    name = "sumosum"
    description = "Text summarization with climate corpus"
    tags = ["summarization", "climate"]

    TRAIN_RATIO = 0.2
    TITLE_KEY = "Title"
    DOCUMENT_KEY = "Doc_text"

    def __init__(
        self,
        filter_min_length: Optional[int] = None,
        filter_max_length: Optional[int] = None,
        truncate_to_length: Optional[int] = None,
    ):
        """
        Initializes the scenario.
        Args:
            sampling_min_length: Int indicating minimum length for training
                                 documents. Training examples smaller than
                                 sampling_min_length tokens will be filtered out.
                                 Useful for preventing the adapter from sampling
                                 really small documents.
            sampling_max_length: Int indicating maximum length for training
                                 documents. Training examples larger than
                                 sampling_max_length tokens will be filtered out.
                                 Useful for preventing the adapter from
                                 sampling really large documents.
            truncate_to_length: Int indicating the maximum length in tokens to
                            truncate documents. Documents in all splits will be
                            truncated to doc_max_length tokens.
                            NOTE: Whitespace tokenization is used to compute tokens.
        """
        super().__init__()
        self.sampling_min_length = filter_min_length
        self.sampling_max_length = filter_max_length
        self.truncate_to_length = truncate_to_length

    @staticmethod
    def _clean_and_truncate(text: str, max_length: Optional[int] = None) -> str:
        text = re.sub(r"\s+", " ", text)
        return " ".join(text.split()[:max_length])

    def _load_dataset(self, output_path: str) -> Dict[str, Dataset]:
        data_dir = os.path.join(output_path, "data")
        ensure_directory_exists(data_dir)

        source_url = "https://github.com/rahulOmishra/SUMO/raw/main/climate_claims_raw.xlsx"
        source_file = os.path.basename(source_url)
        target_path = os.path.join(data_dir, source_file)
        ensure_file_downloaded(
            source_url=source_url,
            target_path=target_path,
        )

        # Claim_id(int),Claim,Title,Doc_text,Label(bool)
        target_df = pd.read_excel(target_path, skiprows=1)
        target_df = target_df.dropna(subset=[SUMOSumScenario.TITLE_KEY, SUMOSumScenario.DOCUMENT_KEY])
        # Remove carriage return _x000D_ in Excel string
        target_df = target_df.replace({r"_x000D_": ""}, regex=True)
        # target_df = target_df.replace({r"_x([0-9a-fA-F]{4})_": ""}, regex=True)
        # Split randomly (works better than split by order)
        train_df = target_df.sample(frac=SUMOSumScenario.TRAIN_RATIO, random_state=0)
        test_df = target_df.drop(train_df.index).sample(frac=1, random_state=0)

        train_ds = Dataset.from_pandas(train_df, split="train")
        test_ds = Dataset.from_pandas(test_df, split="test")
        return {
            "train": train_ds, "test": test_ds
        }

    def get_instances(self, output_path: str) -> List[Instance]:
        dataset_dict = self._load_dataset(output_path)

        instances: List[Instance] = []

        for split, split_data in dataset_dict.items():
            for example in split_data:
                document = SUMOSumScenario._clean_and_truncate(example[SUMOSumScenario.DOCUMENT_KEY], self.truncate_to_length)
                title = SUMOSumScenario._clean_and_truncate(example[SUMOSumScenario.TITLE_KEY])

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
