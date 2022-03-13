from dataclasses import dataclass
from typing import Optional

from .interactions_processor import InteractionsProcessorSpec


@dataclass(frozen=True)
class InteractionModeSpec:
    """Defines what to do during interaction mode."""

    # Whether we are in interaction mode (defaults to False).
    interaction_mode: bool = False

    # Interactions processor
    interactions_processor_spec: Optional[InteractionsProcessorSpec] = None
