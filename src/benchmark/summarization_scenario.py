from typing import List, Optional

from datasets import load_dataset
from .scenario import Scenario, Instance, Reference, TRAIN_SPLIT, VALID_SPLIT, TEST_SPLIT, CORRECT_TAG


class SummarizationScenario(Scenario):
    """
        Scenario for single document text summarization.
        Currently supports the following datasets:
            1. XSum (https://arxiv.org/pdf/1808.08745.pdf)
            2. CNN/DailyMail non-anonymized (https://arxiv.org/pdf/1704.04368.pdf)

        Task prompt structure:
            Summarize the given document.
            Document: {tok_1 ... tok_n}
            Summary: {tok_1 ... tok_m}

        Example from XSum dataset:
            Document: {Part of the Broad Road was closed to traffic on Sunday at about 18:00 GMT.
                       The three adults and three children have been taken to Altnagelvin Hospital
                       with non life-threatening injuries. The Fire Service, Northern Ireland Ambulance Service
                       and police attended the crash. The Broad Road has since been reopened.}
            Summary: {Three adults and three children have been taken to hospital following a crash involving
                      a tractor and a campervan in Limavady, County Londonderry}
    """

    name = "summarization"
    description = "Scenario for summarization tasks"
    tags = ["summarization"]

    def __init__(
        self,
        dataset_name: str,
        sampling_min_length: Optional[int] = None,
        sampling_max_length: Optional[int] = None,
        doc_max_length: Optional[int] = None,
    ):
        """
            Initializes summarization scenario.
            Args:
                dataset_name: String identifier for dataset. Currently
                              supported options ["Xsum", "cnn-dm"].
                sampling_min_length: Int indicating minimum length for training
                                     documents. Training examples smaller than
                                     sampling_min_length will be filtered out.
                                     Useful for preventing the adapter from sampling
                                     really small documents.
                sampling_max_length: Int indicating maximum length for training
                                     documents. Training examples larger than
                                     sampling_max_length will be filtered out.
                                     Useful for preventing the adapter from
                                     sampling really large documents.
                doc_max_length: Int indicating the maximum length to truncate
                                documents. Documents in all splits will be
                                truncated to doc_max_length tokens.
                                NOTE: Currently uses whitespace tokenization.
        """
        if dataset_name not in ["xsum", "cnn-dm"]:
            raise Exception(f"Uknown dataset_name: {dataset_name}")
        self.dataset_name = dataset_name
        self.sampling_min_length = sampling_min_length
        self.sampling_max_length = sampling_max_length
        self.doc_max_length = doc_max_length

    def _clean_and_truncate(self, text: str, max_length: Optional[int] = None):
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
                    if self.sampling_max_length and art_len > self.sampling_max_length:
                        continue
                    if self.sampling_min_length and art_len < self.sampling_min_length:
                        continue

                instances.append(
                    Instance(
                        input=f"{{{article}}}",
                        references=[Reference(output=f"{summary}}}", tags=[CORRECT_TAG])],
                        split=split,
                    )
                )

        return instances
