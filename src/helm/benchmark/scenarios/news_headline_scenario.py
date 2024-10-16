import os
from typing import List

import pandas as pd

from helm.common.general import ensure_directory_exists, ensure_file_downloaded
from helm.benchmark.scenarios.scenario import (
    CORRECT_TAG,
    TEST_SPLIT,
    TRAIN_SPLIT,
    Instance,
    PassageQuestionInput,
    Reference,
    Scenario,
    Output,
)


class NewsHeadlineScenario(Scenario):
    """
    Context:
    This dataset contains the gold commodity news annotated into various dimensions including information such as
    past movements and expected directionality in prices, asset comparison and other general information that the
    news is referring to.
    url: https://www.kaggle.com/datasets/daittan/gold-commodity-news-and-dimensions
    Content:
    The dataset contains 12 columns.

    The data file https://www.kaggle.com/datasets/daittan/gold-commodity-news-and-dimensions?select=finalDataset_0208.csv
    must be downloaded manually and located at {execution_path}/benchmark_output/scenarios/news_headline/restricted/finalDataset_0208.csv.

    Acknowledgements:
    Sinha, Ankur, and Tanmay Khandait.
    "Impact of News on the Commodity Market: Dataset and Results." arXiv preprint arXiv:2009.04202 (2020)
    """  # noqa

    name = "news_headline"
    description = "The dataset is a collection of news items related to the gold commodities from various sources."

    tags = ["news headline", "classification"]

    PROMPT_CATEGORIES = {
        "Price or Not": ["price", "price", "not-price"],
        "Direction Up": ["direction up", "direction-up", "not-direction-up"],
        "Direction Constant": ["direction constant", "direction-constant", "not-direction-constant"],
        "Direction Down": ["direction down", "direction-down", "not-direction-down"],
        "PastPrice": ["past price", "past-price", "not-past-price"],
        "FuturePrice": ["future price", "future-price", "not-future-price"],
        "PastNews": ["past news", "past-news", "not-past-news"],
        "FutureNews": ["future news", "future-news", "not-future-news"],
        "Asset Comparision": ["asset comparison", "asset-comparison", "not-asset-comparison"],  # typo is in the dataset
    }

    def __init__(self, category: str):
        super().__init__()
        assert category in NewsHeadlineScenario.PROMPT_CATEGORIES.keys(), f"Invalid category: {category}"
        self.category: str = category

    def get_instances(self, output_path: str) -> List[Instance]:
        ensure_directory_exists(output_path)
        data_path = os.path.join(output_path, "finalDataset_0208.csv")
        
        ensure_file_downloaded(
            source_url="https://www.kaggle.com/api/v1/datasets/download/daittan/gold-commodity-news-and-dimensions?dataset_version_number=1",
            target_path=data_path,
            unpack=True,
            unpack_type="unzip",
        )

        # read pandas dataframe from csv
        df = pd.read_csv(data_path, index_col=0)

        prompt_question = "Is the passage above about " + NewsHeadlineScenario.PROMPT_CATEGORIES[self.category][0] + "?"

        instances: List[Instance] = []
        for _, row in df.iterrows():
            expected_output: str
            if row[self.category] == 1:
                expected_output = "Yes"
            else:
                expected_output = "No"

            instance = Instance(
                input=PassageQuestionInput(str(row["News"]), prompt_question),
                references=[Reference(Output(text=expected_output), tags=[CORRECT_TAG])],
                # no explicit train/test split, so treat all rows as test cases
                split=str(TEST_SPLIT),
            )
            instances.append(instance)
        return instances
