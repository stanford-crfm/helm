from typing import List

from datasets import load_dataset
from .scenario import Scenario, Instance, Reference, TRAIN_SPLIT, VALID_SPLIT, TEST_SPLIT, CORRECT_TAG


class SummarizationScenario(Scenario):
    """
        Scnario for single document text summarization.
        Currently supports the following datasets:
            1. XSum (https://arxiv.org/pdf/1808.08745.pdf)
            2. CNN/DailyMail non-anonymized (https://arxiv.org/pdf/1704.04368.pdf)
    """

    name = "summarization"
    description = "Scenario for summarization tasks"
    tags = ["summarization"]

    def __init__(
        self,
        dataset_name: str,
        sampling_min_length: int = 0,
        sampling_max_length: int = 50000,
        doc_max_length: int = 50000,
    ):
        assert dataset_name in ["xsum", "cnn-dm"]
        self.dataset_name = dataset_name
        self.sampling_min_length = sampling_min_length
        self.sampling_max_length = sampling_max_length
        self.doc_max_length = doc_max_length

    def _clean_and_truncate(self, text: str, max_length: int = None):
        text = text.replace("\n", " ")
        return " ".join(text.split()[:max_length])

    def _load_dataset(self, dataset_name: str):
        if dataset_name == "xsum":
            dataset = load_dataset("xsum")
            article_key = "document"
            summary_key = "summary"
        elif dataset_name == "cnn-dm":
            dataset = load_dataset("cnn_dailymail", "3.0.0")
            article_key = "article"
            summary_key = "highlights"
        else:
            raise ValueError("The specified dataset is not supported")

        return dataset, article_key, summary_key

    def get_instances(self) -> List[Instance]:
        dataset, article_key, summary_key = self._load_dataset(self.dataset_name)

        splits = {"train": TRAIN_SPLIT, "validation": VALID_SPLIT, "test": TEST_SPLIT}

        instances: List[Instance] = []

        for split_name, split in splits.items():
            for example in dataset[split_name]:
                article = self._clean_and_truncate(example[article_key], self.doc_max_length)
                summary = self._clean_and_truncate(example[summary_key])

                if split_name == "train":
                    art_len = len(article.split())
                    if art_len > self.sampling_max_length or art_len < self.sampling_min_length:
                        continue

                instances.append(
                    Instance(input=article, references=[Reference(output=summary, tags=[CORRECT_TAG])], split=split)
                )

        return instances
