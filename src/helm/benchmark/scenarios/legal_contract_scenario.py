import os
import pandas as pd
import json

from typing import List, Optional
from helm.common.general import ensure_file_downloaded, ensure_directory_exists
from .scenario import Scenario, Instance, Reference, TRAIN_SPLIT, TEST_SPLIT, CORRECT_TAG, PassageQuestionInput, Output


class LegalContractScenario(Scenario):
    """
    Plain English Summarization of Contracts ([paper](https://arxiv.org/abs/1906.00424))
    """

    TRAIN_RATIO: float = 0.2
    ARTICLE_COLUMN_NAME = "original_text"
    SUMMARY_COLUMN_NAME = "reference_summary"

    name = "legal_contract"
    description = "Text summarization with legislative corpus"
    tags = ["summarization", "legal"]

    def __init__(self):
        """
        Initializes the scenario.

        """
        super().__init__()

    @staticmethod
    def _clean(text: str, max_length: Optional[int] = None) -> str:
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

        target_df = pd.DataFrame()
        with open(target_path) as f:
            json_data = json.load(f)
            target_df = pd.DataFrame.from_records(json_data)
            target_df = target_df.dropna(subset=[LegalContractScenario.ARTICLE_COLUMN_NAME, LegalContractScenario.SUMMARY_COLUMN_NAME])
            # Split randomly (works better than split by order)
            train_df = target_df.sample(frac=LegalContractScenario.TRAIN_RATIO, random_state=0)
            test_df = target_df.drop(train_df.index).sample(frac=1, random_state=0)

        return {TRAIN_SPLIT: train_df, TEST_SPLIT: test_df}


    def get_instances(self, output_path: str) -> List[Instance]:
        dataset = self._load_dataset(output_path)

        instances: List[Instance] = []

        PASSAGE_SYNONYM = "text"
        PASSAGE_PREFIX = f"{PASSAGE_SYNONYM.capitalize()}: "
        QUESTION_PREFIX = ""
        QUESTION = f"Write the summary of the above {PASSAGE_SYNONYM}."

        for split, split_data in dataset.items():
            for example in split_data.itertuples():
                article = LegalContractScenario._clean(getattr(example, LegalContractScenario.ARTICLE_COLUMN_NAME))
                summary = LegalContractScenario._clean(getattr(example, LegalContractScenario.SUMMARY_COLUMN_NAME))
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
