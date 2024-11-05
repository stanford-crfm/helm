import os
import pandas as pd
import json
import re

from typing import List
from helm.common.general import ensure_file_downloaded, ensure_directory_exists
from helm.benchmark.scenarios.scenario import (
    Input,
    Scenario,
    Instance,
    Reference,
    TRAIN_SPLIT,
    TEST_SPLIT,
    CORRECT_TAG,
    Output,
)


class LegalContractSummarizationScenario(Scenario):
    """Legal Contract Summarization

    A legal contract summarization benchmark based on the paper
    Plain English Summarization of Contracts (Manor & Li, NAACL 2019),
    which presented a dataset of legal text snippets paired with summaries
    written in plain English.

    @inproceedings{manor-li-2019-plain,
        title = "Plain {E}nglish Summarization of Contracts",
        author = "Manor, Laura  and
        Li, Junyi Jessy",
        editor = "Aletras, Nikolaos  and
        Ash, Elliott  and
        Barrett, Leslie  and
        Chen, Daniel  and
        Meyers, Adam  and
        Preotiuc-Pietro, Daniel  and
        Rosenberg, David  and
        Stent, Amanda",
        booktitle = "Proceedings of the Natural Legal Language Processing Workshop 2019",
        month = jun,
        year = "2019",
        address = "Minneapolis, Minnesota",
        publisher = "Association for Computational Linguistics",
        url = "https://aclanthology.org/W19-2201",
        doi = "10.18653/v1/W19-2201",
        pages = "1--11",
        abstract = "Unilateral legal contracts, such as terms of service, play a substantial role in modern digital life. However, few read these documents before accepting the terms within, as they are too long and the language too complicated. We propose the task of summarizing such legal documents in plain English, which would enable users to have a better understanding of the terms they are accepting. We propose an initial dataset of legal text snippets paired with summaries written in plain English. We verify the quality of these summaries manually, and show that they involve heavy abstraction, compression, and simplification. Initial experiments show that unsupervised extractive summarization methods do not perform well on this task due to the level of abstraction and style differences. We conclude with a call for resource and technique development for simplification and style transfer for legal language.",
    }
    """  # noqa: E501

    TRAIN_RATIO: float = 0.2
    ARTICLE_COLUMN_NAME = "original_text"
    SUMMARY_COLUMN_NAME = "reference_summary"
    ID_COLUMN_NAME = "uid"

    name = "legal_contract_summarization"
    description = (
        "Plain English Summarization of Contracts [(Manor et al., 2019)](https://aclanthology.org/W19-2201.pdf)."
    )
    tags = ["summarization", "legal"]

    def __init__(self):
        """
        Initializes the scenario.

        """
        super().__init__()

    @staticmethod
    def _clean(text: str) -> str:
        return re.sub(r"\s+", " ", text)

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
            target_df = pd.DataFrame.from_records(list(json_data.values()))
            target_df = target_df.dropna(
                subset=[
                    LegalContractSummarizationScenario.ARTICLE_COLUMN_NAME,
                    LegalContractSummarizationScenario.SUMMARY_COLUMN_NAME,
                    LegalContractSummarizationScenario.ID_COLUMN_NAME,
                ]
            )
            # Split randomly (works better than split by order)
            train_df = target_df.sample(frac=LegalContractSummarizationScenario.TRAIN_RATIO, random_state=0)
            test_df = target_df.drop(train_df.index).sample(frac=1, random_state=0)

        return {TRAIN_SPLIT: train_df, TEST_SPLIT: test_df}

    def get_instances(self, output_path: str) -> List[Instance]:
        dataset = self._load_dataset(output_path)

        instances: List[Instance] = []

        for split, split_data in dataset.items():
            for example in split_data.itertuples():
                id = getattr(example, LegalContractSummarizationScenario.ID_COLUMN_NAME)
                article = LegalContractSummarizationScenario._clean(
                    getattr(example, LegalContractSummarizationScenario.ARTICLE_COLUMN_NAME)
                )
                summary = LegalContractSummarizationScenario._clean(
                    getattr(example, LegalContractSummarizationScenario.SUMMARY_COLUMN_NAME)
                )
                input = Input(
                    text=article,
                )
                output = Output(text=summary)
                instance = Instance(
                    id=id,
                    input=input,
                    references=[Reference(output=output, tags=[CORRECT_TAG])],
                    split=split,
                )
                instances.append(instance)

        return instances
