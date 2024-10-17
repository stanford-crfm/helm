from pathlib import Path
import random
import datasets
from typing import Dict, List
from .scenario import Scenario, Instance, Reference, CORRECT_TAG, TRAIN_SPLIT, TEST_SPLIT, Input, Output


def get_instructions():
    instruction = """The dataset consists of sentences from English language financial news categorized by sentiment.
Classify the sentences into one of the 3 sentiment categories.
Possible labels:\n1. positive\n2. neutral\n3. negative"""
    return instruction


SUBSETS = ["sentences_allagree", "sentences_75agree", "sentences_66agree", "sentences_50agree"]


class FinancialPhrasebankScenario(Scenario):
    """
    Context:
    Polar sentiment dataset of sentences from financial news. The dataset consists of 4840 sentences from English
    language financial news categorized by sentiment. The dataset is divided by agreement rate of 5-8 annotators.

    This release of the financial phrase bank covers a collection of 4840 sentences. The selected collection of
    phrases was annotated by 16 people with adequate background knowledge on financial markets.

    Given the large number of overlapping annotations (5 to 8 annotations per sentence), there are several ways
    to define a majority vote based gold standard. To provide an objective comparison, we have formed 4 alternative
    reference datasets based on the strength of majority agreement:

    Data source:
    https://huggingface.co/datasets/takala/financial_phrasebank

    Reference:
    P. Malo, A. Sinha, P. Korhonen, J. Wallenius, and P. Takala, “Good debt or bad debt: Detecting semantic orientations in economic texts,” Journal of the Association for Information Science and Technology, vol. 65, 2014.

    """

    name = "financial_phrasebank"
    description = "The dataset consists of 4840 sentences from English \
                   language financial news categorized by sentiment."
    tags = ["finance", "sentiment analysis", "classification"]

    def __init__(self, subset: str, random_seed: int = 121):
        """The initialization of an instance.

        Args:
            subset: str: This argument is used to specify the ratio of annotators who agreed on the ground truth label. 
            The value must be one of the strings defined in 
            SUBSETS = ["sentences_allagree", "sentences_75agree", "sentences_66agree", "sentences_50agree"].
            random_seed: int = 121: The random seed for sampling the train/test splits.
        """
        super().__init__()
        assert subset in SUBSETS, "Unknown subset: {}".format(subset)
        self.subset = subset
        self.random_seed = random_seed

    def get_instances(self, output_path: str) -> List[Instance]:

        fields = ["sentence"]
        cache_dir = str(Path(output_path) / "data")
        # Download raw data
        # Note: Only using public labeled instances now. Check if we can get the hidden test set labels.
        all_usable_dataset = datasets.load_dataset(
            "financial_phrasebank", self.subset, cache_dir=cache_dir, split="train"
        )
        assert isinstance(all_usable_dataset, datasets.Dataset)
        dataset = all_usable_dataset.train_test_split(train_size=0.8, seed=self.random_seed)
        train_dataset, test_dataset = dataset["train"], dataset["test"]
        class_label_to_string = train_dataset.features["label"].int2str

        dataset_splits: Dict[str, datasets.Dataset] = {
            TRAIN_SPLIT: train_dataset,
            TEST_SPLIT: test_dataset,
        }

        # Read all instances
        random.seed(self.random_seed)
        instances: List[Instance] = []
        for split, subset in dataset_splits.items():
            for x in subset:
                assert fields is not None, "Field ordering not loaded"
                prompt: str = "\n".join([f"{x[field]}" for field in fields])
                instance = Instance(
                    input=Input(text=prompt),
                    references=[Reference(Output(text=class_label_to_string(x["label"])), tags=[CORRECT_TAG])],
                    split=split,
                )
                instances.append(instance)

        return instances
