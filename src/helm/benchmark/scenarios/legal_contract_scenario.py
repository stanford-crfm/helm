import os
from datasets import load_dataset
import pandas as pd
import csv
import json

from typing import List, Optional
from helm.common.general import ensure_file_downloaded, ensure_directory_exists
from .scenario import Scenario, Instance, Reference, TRAIN_SPLIT, TEST_SPLIT, CORRECT_TAG, PassageQuestionInput, Output


class LegalContractScenario(Scenario):
    """
    Plain English Summarization of Contracts ([paper](https://arxiv.org/abs/1906.00424))
    """

    TRAIN_RATIO: float = 0.2

    name = "legal_contract"
    description = "Text summarization with legislative corpus"
    tags = ["summarization", "legal"]

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

        source_url = "https://raw.githubusercontent.com/lauramanor/legal_summarization/master/all_v1.json"
        source_file = os.path.basename(source_url)
        target_path = os.path.join(data_dir, source_file)
        ensure_file_downloaded(
            source_url=source_url,
            target_path=target_path,
        )

        source_file_noext = os.path.splitext(source_file)[0]
        train_file = f"{source_file_noext}-{TRAIN_SPLIT}.csv"
        test_file = f"{source_file_noext}-{TEST_SPLIT}.csv"
        article_key = "original_text"
        summary_key = "reference_summary"
        target_df = pd.DataFrame()
        with open(target_path) as f:
            orig_df = json.load(f)
            for _, dict in orig_df.items():
                target_df = pd.concat([target_df, pd.DataFrame([dict])], ignore_index=True)
            target_df = target_df.dropna(subset=[article_key, summary_key])
            # Split randomly (works better than split by order)
            train_df = target_df.sample(frac=LegalContractScenario.TRAIN_RATIO, random_state=0)
            test_df = target_df.drop(train_df.index).sample(frac=1, random_state=0)
            train_df.to_csv(os.path.join(data_dir, train_file), index=False, header=True, quoting=csv.QUOTE_NONNUMERIC)
            test_df.to_csv(os.path.join(data_dir, test_file), index=False, header=True, quoting=csv.QUOTE_NONNUMERIC)

        data_files = {}
        data_files[TRAIN_SPLIT] = train_file
        data_files[TEST_SPLIT] = test_file
        dataset = load_dataset(data_dir, data_files=data_files)

        return dataset, article_key, summary_key

    def get_instances(self, output_path: str) -> List[Instance]:
        dataset, article_key, summary_key = self._load_dataset(output_path)

        instances: List[Instance] = []

        PASSAGE_SYNONYM = "text"
        PASSAGE_PREFIX = f"{PASSAGE_SYNONYM.capitalize()}: "
        QUESTION_PREFIX = ""
        QUESTION = f"Write the summary of the above {PASSAGE_SYNONYM}."

        for split, split_data in dataset.items():
            for example in split_data:
                article: str = LegalContractScenario._clean_and_truncate(example[article_key], self.doc_max_length)

                if split == TRAIN_SPLIT:
                    art_len = len(article.split())
                    if self.sampling_max_length and art_len > self.sampling_max_length:
                        continue
                    if self.sampling_min_length and art_len < self.sampling_min_length:
                        continue

                summary: str = LegalContractScenario._clean_and_truncate(example[summary_key])

                input = PassageQuestionInput(
                    passage=article,
                    question=QUESTION,
                    passage_prefix=PASSAGE_PREFIX,
                    question_prefix=QUESTION_PREFIX,
                )
                output = Output(text=summary)

                instance = Instance(
                    input=input,
                    references=[Reference(output=output, tags=[CORRECT_TAG])],
                    split=split,
                )
                instances.append(instance)

        return instances
