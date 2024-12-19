from typing import Any, List, Dict
from pathlib import Path
from datasets import load_dataset
from helm.common.hierarchical_logger import hlog
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


class TweetSentBRScenario(Scenario):
    """
    TweetSentBR is a corpus of Tweets in Brazilian Portuguese. It was labeled by several
    annotators following steps stablished on the literature for improving reliability on
    the task of Sentiment Analysis. Each Tweet was annotated in one of the three following classes:

    Positive - tweets where a user meant a positive reaction or evaluation about the main topic on the post;
    Negative - tweets where a user meant a negative reaction or evaluation about the main topic on the post;
    Neutral - tweets not belonging to any of the last classes, usually not making a point, out of topic,
    irrelevant, confusing or containing only objective data.

    This dataset is a subset of the tweetSentBR, it contains only 75 samples from the training set
    and all 2.000+ instances of the test set. This is meant for evaluating language models in a few-shot setting.
    """

    name = "simple_classification"
    description = "Classify tweets into Positive, Negative or Neutral."
    tags = ["classification"]

    def process_dataset(self, dataset: Any, split: str) -> List[Instance]:
        instances: List[Instance] = []
        label_names = {"Positive": "Positivo", "Negative": "Negativo", "Neutral": "Neutro"}
        for example in dataset[split]:
            input = Input(text=example["sentence"])
            # NOTE: For classification scenarios, the reference outputs should be the same
            # for all instances, and should include both correct and incorrect classes.
            # HELM only supports single-label classification. Exactly one reference
            # should have the CORRECT_TAG tag.
            references = [
                Reference(Output(text=label_names[example["label"]]), tags=[CORRECT_TAG]),
            ]
            instance = Instance(input=input, references=references, split=split)
            instances.append(instance)
        return instances

    def get_instances(self, output_path: str) -> List[Instance]:
        instances: List[Instance] = []
        cache_dir = str(Path(output_path) / "data")
        dataset = load_dataset("eduagarcia/tweetsentbr_fewshot", cache_dir=cache_dir)
        splits: Dict[str, str] = {
            "train": TRAIN_SPLIT,
            "test": TEST_SPLIT,
        }
        for split in splits:
            if split not in splits.keys():
                hlog(f"{split} split doesn't exist, skipping")
                continue
            instances.extend(self.process_dataset(dataset, splits[split]))

        return instances
