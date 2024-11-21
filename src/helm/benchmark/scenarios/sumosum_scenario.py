import os
import re
from typing import Dict, List, Optional

import pandas as pd

from helm.common.general import ensure_file_downloaded, ensure_directory_exists
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    TRAIN_SPLIT,
    TEST_SPLIT,
    CORRECT_TAG,
    Input,
    Output,
)

try:
    # Needed for pandas.read_excel
    import openpyxl  # noqa
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, ["ibm-enterprise-scenarios"])


class SUMOSumScenario(Scenario):
    """SUMO Web Claims Summarization

    SUMO Web Claims Summarization is a summarization task over the climate subset from the SUMO dataset.
    The task is to write a title based on the article contents.

    Citation:
    @inproceedings{mishra-etal-2020-generating,
        title = "Generating Fact Checking Summaries for Web Claims",
        author = "Mishra, Rahul  and
        Gupta, Dhruv  and
        Leippold, Markus",
        editor = "Xu, Wei  and
        Ritter, Alan  and
        Baldwin, Tim  and
        Rahimi, Afshin",
        booktitle = "Proceedings of the Sixth Workshop on Noisy User-generated Text (W-NUT 2020)",
        month = nov,
        year = "2020",
        address = "Online",
        publisher = "Association for Computational Linguistics",
        url = "https://aclanthology.org/2020.wnut-1.12",
        doi = "10.18653/v1/2020.wnut-1.12",
        pages = "81--90",
        abstract = "We present SUMO, a neural attention-based approach that learns to establish correctness of textual claims based on evidence in the form of text documents (e.g., news articles or web documents). SUMO further generates an extractive summary by presenting a diversified set of sentences from the documents that explain its decision on the correctness of the textual claim. Prior approaches to address the problem of fact checking and evidence extraction have relied on simple concatenation of claim and document word embeddings as an input to claim driven attention weight computation. This is done so as to extract salient words and sentences from the documents that help establish the correctness of the claim. However this design of claim-driven attention fails to capture the contextual information in documents properly. We improve on the prior art by using improved claim and title guided hierarchical attention to model effective contextual cues. We show the efficacy of our approach on political, healthcare, and environmental datasets.",
    }
    """  # noqa: E501

    name = "sumosum"
    description = "Text summarization with climate corpus"
    tags = ["summarization", "climate"]

    TRAIN_RATIO = 0.2
    TITLE_KEY = "Title"
    DOCUMENT_KEY = "Doc_text"

    def __init__(
        self,
        train_filter_min_length: Optional[int] = None,
        train_filter_max_length: Optional[int] = None,
        test_filter_min_length: Optional[int] = None,
        test_filter_max_length: Optional[int] = None,
        truncate_length: Optional[int] = None,
    ):
        """
        Initializes the scenario.
        Args:
            train_filter_min_length: Int indicating minimum length for training
                                     documents. Train examples smaller than
                                     train_filter_min_length tokens will be filtered out.
            train_filter_max_length: Int indicating maximum length for training
                                     documents. Train examples larger than
                                     train_filter_max_length tokens will be filtered out.
            test_filter_min_length: Int indicating minimum length for training
                                    documents. Test examples smaller than
                                    test_filter_min_length tokens will be filtered out.
            test_filter_max_length: Int indicating maximum length for training
                                    documents. Test examples larger than
                                    test_filter_max_length tokens will be filtered out.
            truncate_length: Int indicating the maximum length in tokens to
                            truncate documents. Documents in all splits will be
                            truncated to truncate_length tokens.
                            NOTE: Whitespace tokenization is used to compute tokens.
        """
        super().__init__()
        self.train_filter_min_length = train_filter_min_length
        self.train_filter_max_length = train_filter_max_length
        self.test_filter_min_length = test_filter_min_length
        self.test_filter_max_length = test_filter_max_length
        self.truncate_length = truncate_length

    @staticmethod
    def _clean_and_truncate(text: str, max_length: Optional[int] = None) -> str:
        text = re.sub(r"\s+", " ", text)
        return " ".join(text.split()[:max_length])

    def _load_dataset(self, output_path: str) -> Dict[str, pd.DataFrame]:
        data_dir = os.path.join(output_path, "data")
        ensure_directory_exists(data_dir)

        source_url = "https://github.com/rahulOmishra/SUMO/raw/main/climate_claims_raw.xlsx"
        source_file = os.path.basename(source_url)
        target_path = os.path.join(data_dir, source_file)
        ensure_file_downloaded(
            source_url=source_url,
            target_path=target_path,
        )

        # Column headers: Claim_id(int),Claim,Title,Doc_text,Label(bool)
        target_df = pd.read_excel(target_path, skiprows=1)
        target_df = target_df.dropna(subset=[SUMOSumScenario.TITLE_KEY, SUMOSumScenario.DOCUMENT_KEY])
        # Remove carriage return _x000D_ in Excel string
        target_df = target_df.replace({r"_x000D_": ""}, regex=True)
        # target_df = target_df.replace({r"_x([0-9a-fA-F]{4})_": ""}, regex=True)
        # Split randomly (works better than split by order)
        train_df = target_df.sample(frac=SUMOSumScenario.TRAIN_RATIO, random_state=0)
        test_df = target_df.drop(train_df.index).sample(frac=1, random_state=0)
        return {TRAIN_SPLIT: train_df, TEST_SPLIT: test_df}

    def get_instances(self, output_path: str) -> List[Instance]:
        dataset_dict = self._load_dataset(output_path)

        instances: List[Instance] = []

        for split, split_data in dataset_dict.items():
            for example in split_data.itertuples():
                document = getattr(example, SUMOSumScenario.DOCUMENT_KEY)
                title = getattr(example, SUMOSumScenario.TITLE_KEY)
                art_len = len(document.split())
                if split == TEST_SPLIT:
                    if self.test_filter_max_length and art_len > self.test_filter_max_length:
                        continue
                    if self.test_filter_min_length and art_len < self.test_filter_min_length:
                        continue
                if split == TRAIN_SPLIT:
                    if self.train_filter_max_length and art_len > self.train_filter_max_length:
                        continue
                    if self.train_filter_min_length and art_len < self.train_filter_min_length:
                        continue

                document = SUMOSumScenario._clean_and_truncate(document, self.truncate_length)
                title = SUMOSumScenario._clean_and_truncate(title)

                instance = Instance(
                    input=Input(text=document),
                    references=[Reference(output=Output(text=title), tags=[CORRECT_TAG])],
                    split=split,
                )
                instances.append(instance)

        return instances
