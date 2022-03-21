from dataclasses import dataclass
from typing import List

from .interactions_processor import InteractionsProcessor
from .interaction_outcome import InteractionOutcome

"""Used for debugging interaction mode."""


@dataclass(frozen=True)
class SimpleConversationOutcome(InteractionOutcome):
    # Unique id of the conversation
    id: str

    # The conversation as a list of strings
    conversation: List[str]


class SimpleInteractionsProcessor(InteractionsProcessor):
    def __init__(self):
        # Nothing to do here since we're not reading from generated files.
        pass

    def process(self) -> List[InteractionOutcome]:
        return [
            SimpleConversationOutcome(id="conversation0", conversation=["Hi", "Hello", "Bye", "Goodbye"]),
            SimpleConversationOutcome(id="conversation1", conversation=["Thank you", "No problem"]),
        ]
