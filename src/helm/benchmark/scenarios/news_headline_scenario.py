import os
from typing import List
import pandas as pd

from helm.common.general import ensure_directory_exists

from .scenario import (
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
        """ The initialization of an instance.

        Args:
            category: str: The category of the news data. This must be one of the strings in PROMPT_CATEGORIES.keys() (see above).

        """
        super().__init__()
        assert category in NewsHeadlineScenario.PROMPT_CATEGORIES.keys(), f"Invalid category: {category}"
        self.category: str = category

    def get_instances(self, output_path: str) -> List[Instance]:
        # data_dir = os.path.join(output_path, "data")
        restricted_dir = os.path.join(output_path, "restricted") # https://crfm-helm.readthedocs.io/en/latest/benchmark/#running-restricted-benchmarks
        # ensure_directory_exists(data_dir)
        ensure_directory_exists(restricted_dir)

        target_path = os.path.join(restricted_dir, "finalDataset_0208.csv")

        # read pandas dataframe from csv
        df = pd.read_csv(target_path, index_col=0)

        # no explicit train/test split, so treat all rows as test cases
        df["split"] = TEST_SPLIT
        df.loc[0:100, "split"] = TRAIN_SPLIT

        prompt_question = "Is the passage above about " + NewsHeadlineScenario.PROMPT_CATEGORIES[self.category][0] + "?"

        instances: List[Instance] = []
        for _, row in df.iterrows():
            expected_output: str
            # sub_split: str
            if row[self.category] == 1:
                expected_output = "Yes"
                # sub_split = NewsHeadlineScenario.PROMPT_CATEGORIES[self.category][1]
            else:
                expected_output = "No"
                # sub_split = NewsHeadlineScenario.PROMPT_CATEGORIES[self.category][2]

            instance = Instance(
                input=PassageQuestionInput(str(row["News"]), prompt_question),
                references=[Reference(Output(text=expected_output), tags=[CORRECT_TAG])],
                split=str(row["split"]),
                # sub_split=sub_split,
            )
            instances.append(instance)
        return instances
