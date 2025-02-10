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


class IMDB_PTBRScenario(Scenario):
    """
    The IMDB dataset is a widely-used benchmark dataset for natural language processing (NLP)
    particularly for text classification and sentiment analysis.
    This is a translated version that is meant to evaluate PT-BR models.
    It consists of movie reviews from the Internet Movie Database (IMDB) and
    includes both positive and negative sentiments labeled for supervised learning.
    """

    name = "simple_classification"
    description = "Classify movie reviews between positive or negative."
    tags = ["classification"]

    def process_dataset(self, dataset: Any, split: str) -> List[Instance]:
        instances: List[Instance] = []
        label_names = {0: "negativo", 1: "positivo"}
        for example in dataset[split]:
            input = Input(text=example["text"])
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
        dataset = load_dataset("maritaca-ai/imdb_pt", cache_dir=cache_dir)
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
