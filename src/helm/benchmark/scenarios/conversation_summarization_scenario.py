import os
import pandas as pd 
from typing import List, Optional
from helm.common.general import ensure_file_downloaded, ensure_directory_exists
from .scenario import Scenario, Instance, Reference, TRAIN_SPLIT, VALID_SPLIT, TEST_SPLIT, CORRECT_TAG, Input, Output

class ConversationSummarizationScenario(Scenario):
    """
    Scenario for single conversation summarization.
    Currently supports the following datasets:
    1. Dialogsum Dataset (https://huggingface.co/datasets/knkarthick/dialogsum)
    2. Samsum Dataset (https://huggingface.co/datasets/samsum)

    Task prompt structure
        Summarize the given conversation.
        conversation: {tok_1 ... tok_n}
        Summary: {tok_1 ... tok_m}

    """

    name = "summarization"
    description = "Scenario for conversation summarization tasks"
    tags = ["summarization"]

    def __init__(
        self,
        dataset_name: str,
        sampling_min_length: Optional[int] = None,
        sampling_max_length: Optional[int] = None,
        conversation_max_length: Optional[int] = None,
    ):
        """
        Initializes summarization scenario.
        Args:
            dataset_name: String identifier for dataset. Currently
                          supported options ["Dialogsum", "Samsum"].
            sampling_min_length: Int indicating minimum length for training
                                 conversations. Training examples smaller than
                                 sampling_min_length will be filtered out.
                                 Useful for preventing the adapter from sampling
                                 really small conversations.
            sampling_max_length: Int indicating maximum length for training
                                 conversation. Training examples larger than
                                 sampling_max_length will be filtered out.
                                 Useful for preventing the adapter from
                                 sampling really large conversation.
            doc_max_length: Int indicating the maximum length to truncate
                            conversation. Conversation in all splits will be
                            truncated to doc_max_length tokens.
                            NOTE: Currently uses whitespace tokenization.
        """
        super().__init__()
        if dataset_name not in ["dialogsum", "samsum"]:
            raise Exception(f"Unknown dataset_name: {dataset_name}")
        self.dataset_name = dataset_name
        self.sampling_min_length = sampling_min_length
        self.sampling_max_length = sampling_max_length
        self.conversation_max_length = conversation_max_length

    def _clean_and_truncate(self, text: str) -> str:
        #to do 
        return text

    def _load_dataset(self, dataset_name: str):
        if dataset_name == "dialogsum":
            dataset = pd.read_csv("/home/toshishjawale/conversation-summary-dataset/dialogsum.csv")
            conversation_key = "dialogue"
            summary_key = "summary"
        elif dataset_name == "samsum":
            dataset = pd.read_csv("/home/toshishjawale/conversation-summary-dataset/Samsum.csv")
            conversation_key = "dialogue"
            summary_key = "summary"
        else:
            raise ValueError("The specified dataset is not supported")

        return dataset, conversation_key, summary_key

    def get_instances(self) -> List[Instance]:
        dataset, conversation_key, summary_key = self._load_dataset(self.dataset_name)

        splits = {"train": TRAIN_SPLIT, "validation": VALID_SPLIT, "test": TEST_SPLIT}

        instances: List[Instance] = []

        for split_name, split in splits.items():
            dataset_split = dataset[dataset['split'] == split]
            for idx in range(len(dataset_split)):
                conversation: str = self._clean_and_truncate(dataset_split[conversation_key].values[idx])
                summary: str = self._clean_and_truncate(dataset_split[summary_key].values[idx])

                instances.append(
                    Instance(
                        input=Input(text=conversation),
                        references=[Reference(Output(text=summary), tags=[CORRECT_TAG])],
                        split=split,
                    )
                )

        return instances
