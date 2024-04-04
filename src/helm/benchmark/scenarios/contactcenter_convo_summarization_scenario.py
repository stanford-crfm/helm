import os
import pickle

from typing import List, Optional
from helm.common.general import ensure_file_downloaded, ensure_directory_exists
from .scenario import Scenario, Instance, Reference, TRAIN_SPLIT, VALID_SPLIT, TEST_SPLIT, CORRECT_TAG, Input, Output


class ContactCenterConvoSummarizationScenario(Scenario):
    """
    Scenario for contact center conversational summarization.
    Currently supports the following datasets:
    1. [TBD]

    Task structure

        Conversation:
                agent: message1
                visitor: message2
                ....
        Summary:


    Example from the dataset
        convo: Part of the Broad Road was closed to traffic on Sunday at about 18:00 GMT.
                   The three adults and three children have been taken to Altnagelvin Hospital
                   with non life-threatening injuries. The Fire Service, Northern Ireland Ambulance Service
                   and police attended the crash. The Broad Road has since been reopened.
        Summary: Three adults and three children have been taken to hospital following a crash involving
                  a tractor and a campervan in Limavady, County Londonderry
    """

    name = "cc_convo_summarization"
    description = "Scenario for contact centern summarization tasks"
    tags = ["cc_convo_summarization"]

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
            dataset_name: String identifier for dataset.
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
        if dataset_name not in ["cresta"]:
            raise Exception(f"Unknown dataset_name: {dataset_name}")
        self.dataset_name = dataset_name
        self.sampling_min_length = sampling_min_length
        self.sampling_max_length = sampling_max_length
        self.doc_max_length = doc_max_length

    def _clean_and_truncate(self, text: str, max_length: Optional[int] = None) -> str:
        text = text.replace("\n", " ")
        return " ".join(text.split()[:max_length])

    def _filter(self, convo: str, summary: str):
        convo_len = len(convo.split())
        if convo_len <= 10:
            return True
        return False

    def _download_dataset(self, url, tag, output_path: str):
        pass

    def _load_dataset(self, dataset_name: str, output_path: str):
        url = ""
        dataset = self._download_dataset(url, "cresta", output_path)
        convo_key = "convo"
        summary_key = "summary"

        return dataset, convo_key, summary_key

    def get_instances(self, output_path: str) -> List[Instance]:
        dataset, convo_key, summary_key = self._load_dataset(self.dataset_name, output_path)

        splits = {"train": TRAIN_SPLIT, "validation": VALID_SPLIT, "test": TEST_SPLIT}

        instances: List[Instance] = []

        for split_name, split in splits.items():
            for example in dataset[split_name]:
                convo: str = self._clean_and_truncate(example[convo_key], self.doc_max_length)
                summary: str = self._clean_and_truncate(example[summary_key])

                if split_name == "train":
                    art_len = len(convo.split())
                    if self.sampling_max_length and art_len > self.sampling_max_length:
                        continue
                    if self.sampling_min_length and art_len < self.sampling_min_length:
                        continue
                    if self.dataset_name == "cresta":
                        if self._filter(convo, summary):
                            continue

                instances.append(
                    Instance(
                        input=Input(text=convo),
                        references=[Reference(Output(text=summary), tags=[CORRECT_TAG])],
                        split=split,
                    )
                )

        return instances
