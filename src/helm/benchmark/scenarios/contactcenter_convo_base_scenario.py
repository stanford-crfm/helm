"""
Base scenario for the contact center conversation.

This scenario defines the basic input structure of the conversation
and shared functions for the contact center conversation scenarios.

The conversation structure is json format with the following fields:
[
    {
        "conversation_name": "cresta-helm-cc-2018-01-12T19:31:31.404000",
        "body": [
            {
                "id": "abcd-1",
                "text": "Hello?",
                "speaker_role": "visitor",
            },
            {
                "id": "abcd-2",
                "text": "Thank you for contacting xxx! My name is Jack. I am here to help you. How can I help you today?",
                "speaker_role": "agent",
            }
        ],
    },
    ...
]
"""

import json
from .scenario import Scenario

class ContactCenterConversationScenario(Scenario):
    """Base scenario for the contact center conversation."""
    name = "cc_conversation"
    description = "Base scenario for contact center conversation tasks"
    tags = ["cc_conversation"]

    def __init__(self, dataset_path: str) -> None:
        """
        Initializes contact center base scenario.
        Args:
            dataset_path: path of dataset to load from.
        """
        super().__init__()
        self.dataset_path = dataset_path

    def _load_conversations(self, dataset_path):
        """
        Load the conversations from the given path.

        Only returns the raw dictionary of conversations, where specific input/output text formatting
        is handled by the subclass scenario.
        Args:
            dataset_path: path of dataset to load from.
        Returns:
            dataset: List of conversation dictionaries.
        """
        with open(dataset_path, 'r', encoding='utf-8') as f:
            raw_chats = json.load(f)
        return raw_chats
