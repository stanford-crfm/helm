import os
from typing import List

import pandas as pd

from helm.common.general import ensure_directory_exists, ensure_file_downloaded
from helm.benchmark.scenarios.scenario import (
    CORRECT_TAG,
    TEST_SPLIT,
    Input,
    Instance,
    Reference,
    Scenario,
    Output,
)



class NewsHeadlineScenario(Scenario):
    """Gold commodity news headline classification

    Context:
    This dataset contains the gold commodity news annotated into various dimensions including information such as
    past movements and expected directionality in prices, asset comparison and other general information that the
    news is referring to.
    url: https://www.kaggle.com/datasets/daittan/gold-commodity-news-and-dimensions
    Content:
    The dataset contains 12 columns.

    Citation:
    Sinha, Ankur, and Tanmay Khandait.
    "Impact of News on the Commodity Market: Dataset and Results." arXiv preprint arXiv:2009.04202 (2020)
    https://arxiv.org/abs/2009.04202"""

    name = "news_headline"
    description = "The dataset is a collection of news items related to the gold commodities from various sources."

    tags = ["news headline", "classification"]

    CATEGORY_COLUMN_NAMES = {
        "price_or_not": "Price or Not",
        "direction_up": "Direction Up",
        "direction_constant": "Direction Constant",
        "direction_down": "Direction Down",
        "past_price": "PastPrice",
        "future_price": "FuturePrice",
        "past_news": "PastNews",
        "future_news": "FutureNews",
        "assert_comparison": "Asset Comparision",
    }

    CATEGORY_INSTRUCTIONS = {
        "price_or_not": "the gold price",
        "direction_up": "the gold price heading up",
        "direction_constant": "the price remaining constant or stable",
        "direction_down": "the gold price heading down",
        "past_price": "any past information about gold prices",
        "future_price": "any future information about gold prices",
        "past_news": "any past information other than the gold prices",
        "future_news": "any future information other than the gold prices",
        "assert_comparison": "a comparison purely in the context of the gold commodity with another asset",
    }

    @classmethod
    def get_instructions(cls, category: str):
        if category not in NewsHeadlineScenario.CATEGORY_INSTRUCTIONS:
            raise ValueError(f"Invalid category: '{category}' Valid categories are: {list(NewsHeadlineScenario.CATEGORY_INSTRUCTIONS.keys())}")
        
        return f"The following are news headlines about the gold commodity. Classify whether the news headline discusses {NewsHeadlineScenario.CATEGORY_INSTRUCTIONS[category]}. Answer only \"Yes\" or \"No\"."

    def __init__(self, category: str):
        super().__init__()
        if category not in NewsHeadlineScenario.CATEGORY_INSTRUCTIONS:
            raise ValueError(f"Invalid category: '{category}' Valid categories are: {list(NewsHeadlineScenario.CATEGORY_INSTRUCTIONS.keys())}")
        self.column_name = NewsHeadlineScenario.CATEGORY_COLUMN_NAMES[category]

    def get_instances(self, output_path: str) -> List[Instance]:
        ensure_directory_exists(output_path)
        data_path = os.path.join(output_path, "finalDataset_0208.csv")
        
        ensure_file_downloaded(
            source_url="https://www.kaggle.com/api/v1/datasets/download/daittan/gold-commodity-news-and-dimensions?dataset_version_number=1",
            target_path=data_path,
            unpack=True,
            unpack_type="unzip",
        )

        df = pd.read_csv(data_path)

        instances: List[Instance] = []
        for _, row in df.iterrows():
            expected_output: str
            if row[self.column_name] == 1:
                expected_output = "Yes"
            else:
                expected_output = "No"

            instance = Instance(
                input=Input(text=str(row["News"])),
                references=[Reference(Output(text=expected_output), tags=[CORRECT_TAG])],
                # no explicit train/test split, so treat all rows as test cases
                split=str(TEST_SPLIT),
            )
            instances.append(instance)
        return instances
