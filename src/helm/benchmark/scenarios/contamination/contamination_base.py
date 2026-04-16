from abc import ABC, abstractmethod
from typing import List

from helm.benchmark.scenarios.scenario import Instance


class ContaminationStrategy(ABC):
    """
    Base class for contamination strategies.
    Each strategy knows how to transform original instances into contaminated instances.
    """

    STRATEGY_NAME: str
    STRATEGY_DISPLAY_NAME: str

    def __init__(self):
        self.language = "en"

    @abstractmethod
    def transform_instances(self, instances: List[Instance]) -> List[Instance]:
        """
        Transforms original instances by applying the masking strategy.
        Returns new instances with modified text.
        """
        pass
