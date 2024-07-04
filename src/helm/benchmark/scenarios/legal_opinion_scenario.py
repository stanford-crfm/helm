import os
from typing import List
import pandas as pd
from helm.common.general import ensure_file_downloaded, ensure_directory_exists
from .scenario import (
    Scenario,
    Instance,
    Reference,
    TRAIN_SPLIT,
    TEST_SPLIT,
    CORRECT_TAG,
    Input,
    Output,
)


class LegalOpinionScenario(Scenario):
    """
    Task:
    - This task is a sentiment classification, which classifies legal phrases as positive, neutral, or negative.

    Dataset:
    - train.xlsx and test.xlsx
        - from https://osf.io/zwhm8/
        - Theses datasets are considered as TRAIN_SPLIT, TEST_SPLIT.
        - Each dataset is excel .xlsx file
            - train.xlsx : [Phrase, Label]
            - test.xlsx  : [sentence, label]
        - Labels
            - label = 0 means negative sentiment
            - label = 1 means neutral sentiment
            - label = 2 means positive sentiment

    Prompt:
        ------
        Classify the following legal phrases as positive, neutral or negative.  (Instructions)
        ...
        ...
        Phrase <Example_TEXT>           (Example phrase)
        Sentiment <Example_OUTPUT>      (Correct sentiment for the example phrase)
        ...
        ...
        Phrase: <TEXT>                  (Target phrase)
        Sentiment: <OUTPUT>             (Output ::= negative | neutral | positive)
        ----
    """

    # Names of the tasks we support

    name = "legal_opinion"
    description = "Predicting the sentiment of the legal text in the positive, negative, or neutral."
    tags = ["classification", "sentiment analysis", "legal"]

    # Constants

    ANSWER_POSITIVE = "positive"
    ANSWER_NEGATIVE = "negative"
    ANSWER_NEUTRAL = "neutral"

    # Methods

    def __init__(self):
        super().__init__()
        self.train_fname = "train.xlsx"
        self.test_fname = "test.xlsx"
        self.url_fname = {
            "https://osf.io/download/hfn62/": self.train_fname,
            "https://osf.io/download/q4adh/": self.test_fname,
        }

    def download_data(self):
        data_dir = self.data_dir
        url_fname = self.url_fname
        ensure_directory_exists(data_dir)
        for url in url_fname:
            file_name = url_fname[url]
            ensure_file_downloaded(
                source_url=url,
                target_path=os.path.join(data_dir, file_name),
                unpack=False,
            )

    def int_senti(self, ilabel: int) -> str:
        if ilabel == 2:
            return self.ANSWER_POSITIVE
        elif ilabel == 0:
            return self.ANSWER_NEGATIVE
        elif ilabel == 1:
            return self.ANSWER_NEUTRAL
        else:
            return self.ANSWER_NEUTRAL

    def create_instances(self, df, split, data_file) -> List[Instance]:
        instances = []
        if data_file == self.train_fname:
            for idx in range(0, len(df)):
                phrase = df.iloc[idx]["Phrase"]
                label = df.iloc[idx]["Label"]
                # convert integer_label to answer_string
                slabel = self.int_senti(int(label))
                input_i = Input(text=phrase)
                # if slabel != self.ANSWER_NEUTRAL:
                reference_i = Reference(Output(text=slabel), tags=[CORRECT_TAG])
                instance = Instance(input_i, [reference_i], split=split)
                instances.append(instance)
            return instances
        elif data_file == self.test_fname:
            for idx in range(0, len(df)):
                phrase = df.iloc[idx]["sentence"]
                label = df.iloc[idx]["label"]
                # convert integer_label to answer_string
                slabel = self.int_senti(int(label))
                input_i = Input(text=phrase)
                reference_i = Reference(Output(text=slabel), tags=[CORRECT_TAG])
                instance = Instance(input_i, [reference_i], split=split)
                instances.append(instance)
            return instances
        else:
            return instances

    def get_instances(self, output_path: str) -> List[Instance]:
        self.data_dir = os.path.join(output_path, "data")

        # download data
        self.download_data()
        # read downloaded xlsx files as dataframe
        data_dir = self.data_dir
        train_file_path = os.path.join(data_dir, self.train_fname)
        test_file_path = os.path.join(data_dir, self.test_fname)
        train_df = pd.read_excel(train_file_path)
        test_df = pd.read_excel(test_file_path)
        # create instances
        train_instances = self.create_instances(train_df, TRAIN_SPLIT, self.train_fname)
        test_instances = self.create_instances(test_df, TEST_SPLIT, self.test_fname)
        all_instances = train_instances + test_instances
        return all_instances
