import os
import random
from typing import List

from helm.common.general import ensure_file_downloaded
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    CORRECT_TAG,
    TRAIN_SPLIT,
    TEST_SPLIT,
    Input,
    Output,
)


class FinancialPhrasebankScenario(Scenario):
    """
    A sentiment classification benchmark based on the dataset from Good Debt or Bad Debt - Detecting Semantic Orientations in Economic Texts [(Malo et al., 2013)](https://arxiv.org/abs/1307.5336).

    Context:
    Polar sentiment dataset of sentences from financial news. The dataset consists of 4840 sentences from English
    language financial news categorized by sentiment. The dataset is divided by agreement rate of 5-8 annotators.

    This release of the financial phrase bank covers a collection of 4840 sentences. The selected collection of
    phrases was annotated by 16 people with adequate background knowledge on financial markets.

    Given the large number of overlapping annotations (5 to 8 annotations per sentence), there are several ways
    to define a majority vote based gold standard. To provide an objective comparison, the paper authors have formed 4 alternative
    reference datasets based on the strength of majority agreement: 100%, 75%, 66% and 50%.

    Data source:
    https://huggingface.co/datasets/takala/financial_phrasebank

    Reference:
    P. Malo, A. Sinha, P. Korhonen, J. Wallenius, and P. Takala, “Good debt or bad debt: Detecting semantic orientations in economic texts,” Journal of the Association for Information Science and Technology, vol. 65, 2014.
    https://arxiv.org/pdf/1307.5336

    """  # noqa: E501

    name = "financial_phrasebank"
    description = "The dataset consists of 4840 sentences from English \
                   language financial news categorized by sentiment."
    tags = ["finance", "sentiment analysis", "classification"]

    INSTRUCTIONS = """The dataset consists of sentences from English language financial news categorized by sentiment.
Classify the sentences into one of the 3 sentiment categories.
Possible labels:\n1. positive\n2. neutral\n3. negative"""  # noqa: E501
    DATASET_URL = "https://huggingface.co/datasets/takala/financial_phrasebank/resolve/598b6aad98f7c8d67be161b12a4b5f2497e07edd/data/FinancialPhraseBank-v1.0.zip"  # noqa: E501
    AGREEMENT_VALUES = [50, 66, 75, 100]
    TRAIN_SPLIT_SIZE = 0.7

    def __init__(self, agreement: int, random_seed: int = 121):
        """The initialization of an instance.

        Args:
            subset: str: This argument is used to specify the ratio of annotators who agreed on the ground truth label.
            The value must be one of the strings defined in
            SUBSETS = ["sentences_allagree", "sentences_75agree", "sentences_66agree", "sentences_50agree"].
            random_seed: int = 121: The random seed for sampling the train/test splits.
        """
        super().__init__()
        if agreement not in self.AGREEMENT_VALUES:
            raise Exception(
                f"Unknown `agreement` value: {agreement}, allowed values are {self.AGREEMENT_VALUES}".format(agreement)
            )
        self.agreement = agreement
        self.random_seed = random_seed

    def get_instances(self, output_path: str) -> List[Instance]:
        data_parent_path = os.path.join(output_path, "data")
        ensure_file_downloaded(
            self.DATASET_URL,
            data_parent_path,
            unpack=True,
            unpack_type="unzip",
        )
        file_name = "Sentences_AllAgree.txt" if self.agreement == 100 else f"Sentences_{self.agreement}Agree.txt"
        data_file_path = os.path.join(data_parent_path, "FinancialPhraseBank-v1.0", file_name)
        with open(data_file_path, mode="r", encoding="iso-8859-1") as f:
            lines = list(f.readlines())
            random.Random(self.random_seed).shuffle(lines)
        train_split_index = int(len(lines) * self.TRAIN_SPLIT_SIZE)
        instances: List[Instance] = []
        for index, line in enumerate(lines):
            sentence, label = line.strip().rsplit("@", 1)
            instance = Instance(
                input=Input(text=sentence),
                references=[Reference(Output(text=label), tags=[CORRECT_TAG])],
                split=TRAIN_SPLIT if index < train_split_index else TEST_SPLIT,
            )
            instances.append(instance)
        return instances
