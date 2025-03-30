from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class REEvalParameters:
    """
    Parameters for reeval evaluation.
    """

    model_ability: Optional[float] = None
    """The inital ability of the model to perform the task. Used for reeval evaluation."""
