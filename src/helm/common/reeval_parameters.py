from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ReevalParameters:
    """
    Parameters for reeval evaluation.
    """

    model_ability: Optional[float] = None
    """The inital ability of the model to perform the task. Used for reeval evaluation."""

    max_samples: Optional[int] = None
    """Maximum number of samples to evaluate in reeval mode"""

    metric_name: Optional[str] = None
    """The main metric name for the scenario. For MMLU, it should be exact_match"""
