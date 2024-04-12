"""
Scenario for contact center conversational summarization.

Loads input conversations defined in the ContactCenterConversationScenario
and packs task specific input/output format for summarization tasks.

Task structure

    Conversation:
            agent: message1
            visitor: message2
            ....
    Summary:
            summary of the conversation


Example from the dataset
    Conversation: 
            agent: hi how can i help you today
            visitor: i need help with my account
            agent: sure what is your account number
            visitor: 123456
            .....
    Summary:
           - agent helped visitor with account number 123456
"""


import json
import os
from typing import List, Optional
from helm.benchmark.scenarios.contactcenter_convo_base_scenario import ContactCenterConversationScenario
from helm.common.general import ensure_file_downloaded, ensure_directory_exists
from .scenario import Instance, Reference, TEST_SPLIT, CORRECT_TAG, Input, Output


class ContactCenterConversationSummarizationScenario(ContactCenterConversationScenario):
    """
    Scenario for contact center conversational summarization.
    """

    name = "cc_convo_summarization"
    description = "Scenario for contact centern summarization tasks"
    tags = ["cc_conversation_summarization"]

    def __init__(
        self,
        dataset_path: str,
        sampling_min_length: Optional[int] = None,
        sampling_max_length: Optional[int] = None,
        doc_max_length: Optional[int] = None,
    ):
        """
        Initializes summarization scenario.
        Args:
            dataset_path: path of dataset to load from
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
        self.dataset_path = dataset_path
        self.sampling_min_length = sampling_min_length
        self.sampling_max_length = sampling_max_length
        self.doc_max_length = doc_max_length

    def _filter(self, convo: str, summary: str):
        """filter on conversation turns"""
        convo_len = len(convo.split('\n'))
        if convo_len <= 10:
            return True
        return False
    
    def _load_summaries(self, dataset_path):
        with open(dataset_path, 'r', encoding='utf-8') as f:
            summaries_list = json.load(f)
        summaries = {item['conversation_name']: item['summary'] for item in summaries_list}
        return summaries

    def get_instances(self) -> List[Instance]:
        conversation_path = os.path.join(self.dataset_path, "conversations.json")
        summary_path = os.path.join(self.dataset_path, "summaries.json")
        conversations = self._load_conversations(conversation_path)
        summaries = self._load_summaries(summary_path)


        instances: List[Instance] = []

        for example in conversations:
            conversation_name = example['conversation_name']
            full_conversation_text =  '\n'.join(f"{item['speaker_role']}:{item['text']}" for item in example['body'])
            summary = summaries[conversation_name]

            # use better tokenization to count tokens
            conversation_len = len(full_conversation_text.split())
            if self.sampling_max_length and conversation_len > self.sampling_max_length:
                continue
            if self.sampling_min_length and conversation_len < self.sampling_min_length:
                continue

            if self._filter(full_conversation_text, summary):
                    continue

            # always load TEST split as we don't offer train data
            instances.append(
                Instance(
                    input=Input(text=full_conversation_text),
                    references=[Reference(Output(text=summary), tags=[CORRECT_TAG])],
                    split=TEST_SPLIT,
                )
            )

        return instances
