from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from .interaction_outcome import InteractionOutcome
from common.object_spec import ObjectSpec, create_object


class InteractionsProcessor(ABC):
    """Processes interactions."""

    @abstractmethod
    def process(self) -> List[InteractionOutcome]:
        pass


@dataclass(frozen=True)
class InteractionsProcessorSpec(ObjectSpec):
    """Defines how to instantiate InteractionsProcessor."""

    pass


def create_interaction_processor(interactions_processor_spec: InteractionsProcessorSpec) -> InteractionsProcessor:
    """Creates an InteractionsProcessor from InteractionsProcessorSpec."""
    return create_object(interactions_processor_spec)
