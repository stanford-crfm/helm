import os
from typing import List

import pandas as pd

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
from helm.common.general import ensure_file_downloaded, ensure_directory_exists


class LegalOpinionSentimentClassificationScenario(Scenario):
    """
        TODO: Fill this in

        Example prompt:
    Classify the sentences into one of the 3 sentiment categories. Possible labels: positive, neutral, negative.
    {Sentence}
    Label: {positive/neutral/negative}

    """

    # Names of the tasks we support

    name = "legal_opinion"
    description = "Predicting the sentiment of the legal text in the positive, negative, or neutral."
    tags = ["classification", "sentiment analysis", "legal"]

    SENTIMENT_CLASSES = ["positive", "negative", "neutral"]
    SPLIT_TO_URL = {
        TRAIN_SPLIT: "https://osf.io/download/hfn62/train.xlsx",
        TEST_SPLIT: "https://osf.io/download/hfn62/test.xlsx",
    }

    def create_instances(self, df: pd.DataFrame, split: str) -> List[Instance]:
        instances: List[Instance] = []
        assert split in [TRAIN_SPLIT, TEST_SPLIT]
        if split == TRAIN_SPLIT:
            phrase_column_name = "Phrase"
            label_column_name = "Label"
        else:
            phrase_column_name = "sentence"
            label_column_name = "label"
        for row in df.itertuples():
            phrase = getattr(row, phrase_column_name)
            label_index = int(getattr(row, label_column_name))
            label = LegalOpinionSentimentClassificationScenario.SENTIMENT_CLASSES[label_index]
            instance = Instance(
                input=Input(text=phrase), references=[Reference(Output(text=label), tags=[CORRECT_TAG])], split=split
            )
            instances.append(instance)
        return instances

    def get_instances(self, output_path: str) -> List[Instance]:
        self.data_dir = os.path.join(output_path, "data")
        data_dir = self.data_dir
        ensure_directory_exists(data_dir)
        instances: List[Instance] = []
        for split, url in LegalOpinionSentimentClassificationScenario.SPLIT_TO_URL.items():
            file_name = url.split("/")[-1]
            file_path = os.path.join(data_dir, file_name)
            ensure_file_downloaded(
                source_url=url,
                target_path=os.path.join(data_dir, file_name),
            )
            df = pd.read_excel(file_path)
            instances.extend(self.create_instances(df, split))
        return instances
