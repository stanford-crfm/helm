import os
import pickle

from typing import List, Optional
from helm.common.general import ensure_file_downloaded, ensure_directory_exists
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    TRAIN_SPLIT,
    VALID_SPLIT,
    TEST_SPLIT,
    CORRECT_TAG,
    Input,
    Output,
)


class SummarizationScenario(Scenario):
    """
    Scenario for single document text summarization.
    Currently supports the following datasets:
    1. XSum (https://arxiv.org/pdf/1808.08745.pdf)
    2. CNN/DailyMail non-anonymized (https://arxiv.org/pdf/1704.04368.pdf)

    Task prompt structure

        Summarize the given document.
        Document: {tok_1 ... tok_n}
        Summary: {tok_1 ... tok_m}

    Example from XSum dataset

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
        super().__init__()
        if dataset_name not in ["xsum", "xsum-sampled", "cnn-dm"]:
            raise Exception(f"Unknown dataset_name: {dataset_name}")
        self.dataset_name = dataset_name
        self.sampling_min_length = sampling_min_length
        self.sampling_max_length = sampling_max_length
        self.doc_max_length = doc_max_length

    def _clean_and_truncate(self, text: str, max_length: Optional[int] = None) -> str:
        text = text.replace("\n", " ")
        return " ".join(text.split()[:max_length])

    def _xsum_filter(self, article: str, summary: str):
        art_len = len(article.split())
        summ_len = len(summary.split())

        if "Media playback is unsupported on your device" in article:
            return True

        if "Last updated at" in article:
            return True

        if summ_len <= 10:
            return True

        if summ_len / art_len > 0.2:
            return True

        return False

    def _download_dataset(self, url, tag, output_path: str):
        data_dir = os.path.join(output_path, "data")
        ensure_directory_exists(data_dir)
        ensure_file_downloaded(source_url=url, target_path=os.path.join(data_dir, f"{tag}.pk"))

        with open(os.path.join(data_dir, f"{tag}.pk"), "rb") as fin:
            dataset = pickle.load(fin)

        return dataset

    def _load_dataset(self, dataset_name: str, output_path: str):
        if dataset_name == "xsum":
            url = (
                "https://storage.googleapis.com/crfm-helm-public/source_datasets/"
                "scenarios/summarization_scenario/xsum.pk"
            )
            dataset = self._download_dataset(url, "xsum", output_path)
            article_key = "document"
            summary_key = "summary"
        elif dataset_name == "xsum-sampled":
            url = (
                "https://storage.googleapis.com/crfm-helm-public/source_datasets/"
                "scenarios/summarization_scenario/xsum-sampled.pk"
            )
            dataset = self._download_dataset(url, "xsum-sampled", output_path)
            article_key = "document"
            summary_key = "summary"
        elif dataset_name == "cnn-dm":
            url = (
                "https://storage.googleapis.com/crfm-helm-public/source_datasets/"
                "scenarios/summarization_scenario/cnndm.pk"
            )
            dataset = self._download_dataset(url, "cnndm", output_path)
            article_key = "article"
            summary_key = "highlights"
        else:
            raise ValueError("The specified dataset is not supported")

        return dataset, article_key, summary_key

    def get_instances(self, output_path: str) -> List[Instance]:
        dataset, article_key, summary_key = self._load_dataset(self.dataset_name, output_path)

        splits = {"train": TRAIN_SPLIT, "validation": VALID_SPLIT, "test": TEST_SPLIT}

        instances: List[Instance] = []

        for split_name, split in splits.items():
            for example in dataset[split_name]:
                article: str = self._clean_and_truncate(example[article_key], self.doc_max_length)
                summary: str = self._clean_and_truncate(example[summary_key])

                if split_name == "train":
                    art_len = len(article.split())
                    if self.sampling_max_length and art_len > self.sampling_max_length:
                        continue
                    if self.sampling_min_length and art_len < self.sampling_min_length:
                        continue
                    if self.dataset_name == "xsum":
                        if self._xsum_filter(article, summary):
                            continue

                instances.append(
                    Instance(
                        input=Input(text=article),
                        references=[Reference(Output(text=summary), tags=[CORRECT_TAG])],
                        split=split,
                    )
                )

        return instances
