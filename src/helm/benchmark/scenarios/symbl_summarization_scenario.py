import os
import pandas as pd
from typing import List, Optional
from helm.common.general import ensure_file_downloaded, ensure_directory_exists
from .scenario import Scenario, Instance, Reference, TRAIN_SPLIT, VALID_SPLIT, TEST_SPLIT, CORRECT_TAG, Input, Output

prompt_dict = {"General_List_format/Bullet_point" : "Summarize this conversation in a list or bullet point format.",
        "Sales_List_format/Bullet_point" : "Summarize this conversation in a list or bullet point format.",
        "General_long_summary" : "Summarize this conversation in detail.",
        "Sales_long_summary" : "Summarize this conversation in detail.",
        "General_short_summary":"Give a short summary of this conversation. ",
        "Sales_short_summary":"Give a short summary of this conversation. "
        }

class SymblSummarizationScenario(Scenario):

    """
    Scenario for single conversation summarization.

    Task prompt structure
        Summarize the given conversation.
        conversation: {tok_1 ... tok_n}
        Summary: {tok_1 ... tok_m}

    """

    name = "summarization"
    description = "Scenario for symbl conversation summarization tasks"
    tags = ["summarization"]

    def __init__(
        self,
        summary_type: str,
        sampling_min_length: Optional[int] = None,
        sampling_max_length: Optional[int] = None,
        conversation_max_length: Optional[int] = None,
    ):
        """
        Initializes summarization scenario.
        Args:
            prompt :                  String identifier for prompts. Currently
                                      supported options ["Summarize this conversation in a list or bullet point format."]

            sampling_min_length:      Int indicating minimum length for training
                                      conversations. Training examples smaller than
                                      sampling_min_length will be filtered out.
                                      Useful for preventing the adapter from sampling
                                      really small conversations.

            sampling_max_length:      Int indicating maximum length for training
                                      conversation. Training examples larger than
                                      sampling_max_length will be filtered out.
                                      Useful for preventing the adapter from
                                      sampling really large conversation.

            conversation_max_length:  Int indicating the maximum length to truncate
                                      conversation. Conversation in all splits will be
                                      runcated to doc_max_length tokens.
        """
        super().__init__()
        if summary_type not in list(prompt_dict.keys()):
            raise Exception(f"Unknown summary type: {summary_type}")
        self.summary_type = summary_type
        self.sampling_min_length = sampling_min_length
        self.sampling_max_length = sampling_max_length
        self.conversation_max_length = conversation_max_length

    def _clean_and_truncate(self, text: str) -> str:
        #to do
        return text

    def _load_dataset(self, summary_type: str):
          dataset = pd.read_csv("/home/pratik.budruk/conversation-summary-dataset/Symbl_summarization_all.csv")
          meeting_type = summary_type.split("_")[0]
          dataset = dataset[(dataset['prompt'] == prompt_dict[summary_type]) & (dataset['meeting_type'] == meeting_type)]
          conversation_key = "conversation"
          output_key = "gpt4_output"
          if len(dataset) == 0:
            raise ValueError("The specified prompt is not supported")

          return dataset, conversation_key, output_key

    def get_instances(self) -> List[Instance]:
        dataset, conversation_key, output_key = self._load_dataset(self.summary_type)

        splits = {"train": TRAIN_SPLIT, "validation": VALID_SPLIT, "test": TEST_SPLIT}

        instances: List[Instance] = []

        for split_name, split in splits.items():
            dataset_split = dataset[dataset['split'] == split]
            for idx in range(len(dataset_split)):
                conversation: str = self._clean_and_truncate(dataset_split[conversation_key].values[idx])
                gpt_4_output: str = self._clean_and_truncate(dataset_split[output_key].values[idx])

                instances.append(
                    Instance(
                        input=Input(text=conversation),
                        references=[Reference(Output(text=gpt_4_output), tags=[CORRECT_TAG])],
                        split=split,
                    )
                )

        return instances
